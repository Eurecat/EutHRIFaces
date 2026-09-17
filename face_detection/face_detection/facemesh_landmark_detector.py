#!/usr/bin/env python3
"""
Fast 478-point face landmarks from the YOLO face detections.

Runs only the landmark network of MediaPipe's face_landmarker.task
(face_landmarks_detector, converted to ONNX by tools/convert_face_landmarks_to_onnx.sh into
weights/face_landmarks_detector.onnx) on
ONNX Runtime GPU, all faces of a frame in one batch. MediaPipe's own pipeline also runs
its BlazeFace detector on every crop, one face at a time on CPU, and misses small faces.
Here the region of interest comes from the YOLO box and eye points instead, which is
what MediaPipe does internally with its own detector output (square box, rotated so the
eyes are horizontal).

The output uses the same 478 -> 68 mapping as MediaPipeLandmarkDetector, so consumers
(VSAD, gaze, face recognition alignment) receive the same landmark layout.
"""
import os
import threading
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .mediapipe_landmark_detector import MediaPipeLandmarkDetector

_INPUT_SIZE = 256
_MAP_68 = [MediaPipeLandmarkDetector.MEDIAPIPE_TO_ROS4HRI[i] for i in range(68)]


class FaceMeshOnnxLandmarkDetector:
    """Batched face-mesh landmarks for detector boxes.

    Args:
        model_path: ONNX landmark model (input N x 256 x 256 x 3 RGB in [0, 1]).
        logger: ROS or Python logger.
        device: "cuda..." uses the GPU providers, anything else runs on CPU.
        roi_scale: Side of the square crop as a multiple of the larger YOLO box side.
        roi_shift: Crop centre moved down (along the face axis) by this fraction of the box height.
            1.4 / 0.05 matched MediaPipe best on video_3 (median error 5% of the eye distance).
        min_face_score: Minimum face-presence logit; lower results are dropped (None).
        use_tensorrt: Build a TensorRT FP16 engine in the background (cached next to the model
            weights) and switch to it when ready; CUDA serves meanwhile.
        max_batch: Largest number of faces in one TensorRT batch (bigger batches are split).
    """

    def __init__(self, model_path: str, logger: Any, device: str = "cuda", roi_scale: float = 1.4,
                 roi_shift: float = 0.05, min_face_score: float = 0.0, use_tensorrt: bool = True, max_batch: int = 8,
                 trt_cache_dir: Optional[str] = None):
        self.logger = logger
        self.model_path = model_path
        self.roi_scale = roi_scale
        self.roi_shift = roi_shift
        self.min_face_score = min_face_score
        self.max_batch = max(1, int(max_batch))
        self.session = None
        self._input_name = None
        if not os.path.exists(model_path):
            self.logger.error(f"Face mesh landmark model not found: {model_path}")
            return
        gpu = str(device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self.logger.info(f"Face mesh landmarks (ONNX) ready on {self.session.get_providers()[0]}")
        if gpu and use_tensorrt and "TensorrtExecutionProvider" in ort.get_available_providers():
            self._start_tensorrt_session(trt_cache_dir or os.path.join(os.path.dirname(model_path), "trt_cache"))

    def is_available(self) -> bool:
        return self.session is not None

    _gpu_crops_ok = True

    def _run_cpu(self, image: np.ndarray, affines: List[np.ndarray]):
        crops = [cv2.warpAffine(image, A, (_INPUT_SIZE, _INPUT_SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                 for A in affines]
        stacked = np.stack(crops)
        batch = np.empty(stacked.shape, np.float32)
        np.divide(stacked[..., ::-1], 255.0, out=batch, casting="unsafe")  # BGR -> RGB in [0, 1]
        points, scores = [], []
        for start in range(0, len(batch), self.max_batch):
            out = self.session.run(None, {self._input_name: batch[start:start + self.max_batch]})
            points.append(out[0].reshape(-1, 478, 3)[:, :, :2])
            scores.append(out[1].reshape(-1))
        return np.concatenate(points), np.concatenate(scores)

    def _run_gpu(self, gpu_rgb, affines: List[np.ndarray]):
        """Sample all face crops from the GPU frame with one grid_sample and run the model on them."""
        import torch
        import torch.nn.functional as F

        _, height, width = gpu_rgb.shape
        # affine_grid works in normalised coordinates (align_corners=False): output pixel
        # p_out = 128 * g + 127.5, input normalised g_in = (2 * p_in + 1) / size - 1.
        to_out_pixels = np.array([[128.0, 0, 127.5], [0, 128.0, 127.5], [0, 0, 1]], np.float64)
        to_in_norm = np.array([[2.0 / width, 0, 1.0 / width - 1], [0, 2.0 / height, 1.0 / height - 1], [0, 0, 1]],
                              np.float64)
        thetas = np.stack([(to_in_norm @ np.vstack([A, [0, 0, 1]]) @ to_out_pixels)[:2] for A in affines])
        theta = torch.from_numpy(thetas.astype(np.float32)).to(gpu_rgb.device)
        n = len(affines)
        frame = gpu_rgb.unsqueeze(0).float().div_(255.0)
        grid = F.affine_grid(theta, (n, 3, _INPUT_SIZE, _INPUT_SIZE), align_corners=False)
        crops = F.grid_sample(frame.expand(n, -1, -1, -1), grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)
        batch = crops.permute(0, 2, 3, 1).contiguous()  # N x 256 x 256 x 3, RGB in [0, 1]
        torch.cuda.current_stream(gpu_rgb.device).synchronize()

        points, scores = [], []
        for start in range(0, n, self.max_batch):
            chunk = batch[start:start + self.max_batch].contiguous()
            binding = self.session.io_binding()
            binding.bind_input(self._input_name, device_type="cuda", device_id=gpu_rgb.device.index or 0,
                               element_type=np.float32, shape=tuple(chunk.shape), buffer_ptr=chunk.data_ptr())
            for output in self.session.get_outputs():
                binding.bind_output(output.name)
            self.session.run_with_iobinding(binding)
            out = binding.copy_outputs_to_cpu()
            points.append(out[0].reshape(-1, 478, 3)[:, :, :2])
            scores.append(out[1].reshape(-1))
        return np.concatenate(points), np.concatenate(scores)

    def _start_tensorrt_session(self, cache_dir: str) -> None:
        name = self._input_name

        def build():
            try:
                os.makedirs(cache_dir, exist_ok=True)
                self.logger.info(f"Loading or building TensorRT FP16 engine for face mesh landmarks ({cache_dir}); "
                                 "the first build takes several minutes")
                options = {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": cache_dir,
                    "trt_profile_min_shapes": f"{name}:1x{_INPUT_SIZE}x{_INPUT_SIZE}x3",
                    "trt_profile_opt_shapes": f"{name}:5x{_INPUT_SIZE}x{_INPUT_SIZE}x3",
                    "trt_profile_max_shapes": f"{name}:{self.max_batch}x{_INPUT_SIZE}x{_INPUT_SIZE}x3",
                }
                session = ort.InferenceSession(
                    self.model_path, providers=[("TensorrtExecutionProvider", options), "CUDAExecutionProvider"])
                if "TensorrtExecutionProvider" not in session.get_providers():
                    self.logger.warn("TensorRT unavailable for face mesh landmarks; staying on CUDA")
                    return
                for batch in sorted({1, min(5, self.max_batch), self.max_batch}):
                    session.run(None, {name: np.zeros((batch, _INPUT_SIZE, _INPUT_SIZE, 3), np.float32)})
                self.session = session
                self.logger.info("Face mesh landmarks switched to TensorRT FP16")
            except Exception as e:
                self.logger.warn(f"TensorRT face mesh session failed ({e}); staying on CUDA")

        threading.Thread(target=build, name="facemesh_trt_build", daemon=True).start()

    def _roi_affine(self, bbox: Sequence[float], eyes: np.ndarray) -> np.ndarray:
        """2x3 matrix mapping crop pixels to image pixels (square ROI rotated to level the eyes)."""
        x, y, w, h = bbox
        (lx, ly), (rx, ry) = sorted([tuple(eyes[0]), tuple(eyes[1])])
        angle = np.arctan2(ry - ly, rx - lx)
        c, s = np.cos(angle), np.sin(angle)
        k = max(w, h) * self.roi_scale / _INPUT_SIZE
        half = _INPUT_SIZE / 2.0
        cx = x + w / 2.0 - s * self.roi_shift * h
        cy = y + h / 2.0 + c * self.roi_shift * h
        return np.array([[c * k, -s * k, cx - (c * half - s * half) * k],
                         [s * k, c * k, cy - (s * half + c * half) * k]], dtype=np.float32)

    def detect_landmarks_batch(self, image: np.ndarray, face_bboxes: List[Sequence[float]],
                               yolo_landmarks: List[Sequence[float]],
                               gpu_rgb: Any = None) -> List[Optional[List[Tuple[float, float]]]]:
        """68 (x, y) pixel landmarks per face (ros4hri layout), None when no face is found.

        face_bboxes: (x, y, w, h) per face. yolo_landmarks: 10 values per face
        (left eye, right eye, nose, mouth corners), only the eyes are used for rotation.
        gpu_rgb: optional uint8 3xHxW CUDA tensor of the same frame (from JpegDecoder); crops are
        then sampled on the GPU and handed to ONNX Runtime without CPU copies.
        """
        results: List[Optional[List[Tuple[float, float]]]] = [None] * len(face_bboxes)
        if self.session is None or not face_bboxes:
            return results
        affines, crops, indices = [], [], []
        for i, (bbox, lms) in enumerate(zip(face_bboxes, yolo_landmarks)):
            if lms is None or len(lms) < 4:
                continue
            affines.append(self._roi_affine(bbox, np.asarray(lms[:4], np.float32).reshape(2, 2)))
            indices.append(i)
        if not affines:
            return results
        outputs = None
        if gpu_rgb is not None and self._gpu_crops_ok:
            try:
                outputs = self._run_gpu(gpu_rgb, affines)
            except Exception as e:
                self._gpu_crops_ok = False
                self.logger.warn(f"GPU face crops failed ({e}); using CPU crops")
        if outputs is None:
            outputs = self._run_cpu(image, affines)
        points, scores = outputs
        for i, A, pts, score in zip(indices, affines, points, scores):
            if score < self.min_face_score:
                continue
            image_pts = np.hstack([pts[_MAP_68], np.ones((68, 1), np.float32)]) @ A.T
            results[i] = [(float(px), float(py)) for px, py in image_pts]
        return results
