#!/usr/bin/env python3
"""
YOLO-based face detection implementation for ROS2 face detection node.

This detector uses a YOLOv8-based face detection model with landmarks support.
"""
import os
import math
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# BOXMOT imports
try:
    import boxmot
    from boxmot import create_tracker, get_tracker_config
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False
    print("[WARNING] BOXMOT not available for face tracking. Using simple enumeration instead.")


class YoloFaceDetector:
    """
    YOLOv8-based face detection with facial landmarks.
    
    This detector works on full images and provides:
    - Face bounding boxes
    - 5 facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.2, iou_threshold: float = 0.5, device: str = "cpu", debug: bool = False, use_boxmot: bool = False, boxmot_tracker_type: str = "bytetrack", boxmot_reid_model: str = ""):
        """
        Initialize YOLO face detector.
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for face detection
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cpu" or "cuda")
            debug: Enable debug output
            use_boxmot: Enable BOXMOT tracking
            boxmot_tracker_type: Type of BOXMOT tracker to use
            boxmot_reid_model: Path to ReID model for BOXMOT
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.debug = debug
        
        # BOXMOT tracking parameters
        self.use_boxmot = use_boxmot and BOXMOT_AVAILABLE
        self.boxmot_tracker_type = boxmot_tracker_type
        self.boxmot_reid_model = boxmot_reid_model
        self.boxmot_tracker = None
        
        # Model parameters
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        
        # Model components
        self.net = None
        self.session = None
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), 
                         math.ceil(self.input_width / self.strides[i])) 
                        for i in range(len(self.strides))]
        self.anchors = None
        self.is_initialized = False
        
        # Model download URL
        self.default_model_url = "https://raw.githubusercontent.com/hpc203/yolov8-face-landmarks-opencv-dnn/main/weights/yolov8n-face.onnx"
        
        # Log tracker status
        if self.use_boxmot:
            if BOXMOT_AVAILABLE:
                print(f"[INFO] BOXMOT face tracking enabled with {self.boxmot_tracker_type}")
            else:
                print("[WARNING] BOXMOT requested but not available, falling back to simple enumeration")
        else:
            print("[INFO] Using simple face enumeration (no tracking)")
    
    def _download_model(self, model_path: str, url: str) -> bool:
        """
        Download the YOLO face detection model if it doesn't exist.
        
        Args:
            model_path: Path where to save the model
            url: URL to download the model from
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            print(f"[INFO] Downloading YOLO face model from {url}")
            print(f"[INFO] Saving to {model_path}")
            
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\r[INFO] Download progress: {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, model_path, progress_hook)
            print()  # New line after progress
            
            # Verify the file was downloaded
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                print(f"[INFO] Model downloaded successfully: {os.path.getsize(model_path)} bytes")
                return True
            else:
                print(f"[ERROR] Downloaded file is empty or doesn't exist")
                return False
                
        except urllib.error.URLError as e:
            print(f"[ERROR] Failed to download model from {url}: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error during model download: {e}")
            return False
    
    def initialize(self) -> bool:
        """
        Initialize the YOLO face detection model.
        Auto-downloads the model if it doesn't exist.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Check if model exists, if not try to download it
            if not os.path.exists(self.model_path):
                print(f"[INFO] YOLO face model not found at: {self.model_path}")
                print(f"[INFO] Attempting to auto-download...")
                
                if not self._download_model(self.model_path, self.default_model_url):
                    print(f"[ERROR] Failed to download YOLO face model")
                    return False
            else:
                print(f"[INFO] Found existing YOLO face model at: {self.model_path}")
            
            print(f"[INFO] Initializing YOLO face detector from {self.model_path}")
            print(f"[INFO] Using device: {self.device}")
            
            # Check for onnxruntime-gpu if CUDA is requested
            if 'cuda' in self.device.lower():
                try:
                    if 'CUDAExecutionProvider' not in ort.get_available_providers():
                        print(f"\033[91m[WARNING] onnxruntime-gpu not installed! Install with: pip install onnxruntime-gpu\033[0m")
                except: pass
            
            # Initialize OpenCV DNN (backup method)
            self.net = cv2.dnn.readNet(self.model_path)
            
            # Initialize ONNX InferenceSession with GPU/CPU support
            providers = ['CUDAExecutionProvider'] if 'cuda' in self.device.lower() else ['CPUExecutionProvider']
            
            print(f"[DEBUG] Providers: {providers}")
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Generate anchors
            self.anchors = self._make_anchors(self.feats_hw)
            
            # Initialize BOXMOT tracker if enabled
            if self.use_boxmot:
                self._initialize_boxmot()
            
            self.is_initialized = True
            print(f"[INFO] YOLO face detector initialized successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize YOLO face detector: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Face detection results with keys:
                - 'faces': List of face bounding boxes (x, y, w, h)
                - 'confidences': List of face confidence scores
                - 'landmarks': List of facial landmarks (5 points per face)
        """
        if not self.is_initialized:
            return {"faces": [], "confidences": [], "landmarks": []}
        
        try:
            if self.debug:
                print(f"[DEBUG YOLO] Input image shape: {image.shape}, dtype: {image.dtype}")
            
            # Run face detection on full image
            face_boxes, face_scores, face_classids, face_landmarks = self._detect_faces(image)
            
            if self.debug:
                print(f"[DEBUG YOLO] Raw detection results: {len(face_boxes)} boxes, {len(face_scores)} scores")
                if len(face_scores) > 0:
                    print(f"[DEBUG YOLO] Score range: {min(face_scores):.3f} - {max(face_scores):.3f}")
                    print(f"[DEBUG YOLO] Confidence threshold: {self.conf_threshold}")
            
            # Handle case where detection returns empty results
            if len(face_boxes) == 0:
                if self.debug:
                    print(f"[DEBUG YOLO] No faces detected")
                return {"faces": [], "confidences": [], "landmarks": [], "track_ids": []}
            
            # Apply tracking if BOXMOT is enabled
            track_ids = []
            if self.use_boxmot and self.boxmot_tracker is not None:
                track_ids = self._apply_boxmot_tracking(face_boxes, face_scores, face_classids, image)
            else:
                # Use simple enumeration as fallback
                track_ids = list(range(len(face_boxes)))
                # track_ids = [-1] * len(face_boxes)
            
            # Convert to expected format
            faces = []
            confidences = []
            landmarks = []
            
            for i in range(len(face_boxes)):
                try:
                    # Convert from (x, y, w, h) format and ensure valid coordinates
                    x, y, w, h = face_boxes[i]
                    
                    # Ensure all values are finite and not NaN
                    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(w) and np.isfinite(h)):
                        continue
                    
                    # Clamp bounding box to image bounds
                    x = max(0, int(x))
                    y = max(0, int(y))
                    w = max(1, int(w))
                    h = max(1, int(h))
                    
                    # Ensure bounding box doesn't exceed image boundaries
                    if x >= image.shape[1] or y >= image.shape[0]:
                        continue
                        
                    if x + w > image.shape[1]:
                        w = image.shape[1] - x
                    if y + h > image.shape[0]:
                        h = image.shape[0] - y
                    
                    # Skip if resulting box is too small
                    if w <= 0 or h <= 0:
                        continue
                    
                    faces.append([x, y, w, h])
                    confidences.append(float(face_scores[i]))
                    
                    # Extract landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
                    face_landmarks_5pt = []
                    if i < len(face_landmarks) and len(face_landmarks[i]) >= 15:
                        for j in range(0, 15, 3):  # 15 values = 5 points * 3 coords each (x, y, confidence)
                            lm_x = float(face_landmarks[i][j])
                            lm_y = float(face_landmarks[i][j + 1])
                            
                            # Check for finite values
                            if np.isfinite(lm_x) and np.isfinite(lm_y):
                                lm_x = max(0, min(lm_x, image.shape[1] - 1))
                                lm_y = max(0, min(lm_y, image.shape[0] - 1))
                            else:
                                lm_x, lm_y = 0.0, 0.0
                            
                            face_landmarks_5pt.extend([lm_x, lm_y])
                    else:
                        # If no landmarks available, fill with zeros
                        face_landmarks_5pt = [0.0] * 10
                    
                    landmarks.append(face_landmarks_5pt)
                    
                except Exception as e:
                    print(f"[WARNING] Error processing face {i}: {e}")
                    continue
            
            if self.debug:
                print(f"[DEBUG YOLO] Final results: {len(faces)} faces after processing")
                for i, (face, conf) in enumerate(zip(faces, confidences)):
                    print(f"[DEBUG YOLO] Face {i}: bbox={face}, conf={conf:.3f}")
            
            return {
                "faces": faces,
                "confidences": confidences,
                "landmarks": landmarks,
                "track_ids": track_ids
            }
            
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {"faces": [], "confidences": [], "landmarks": [], "track_ids": []}
    
    def get_detector_type(self) -> str:
        """Get the detector type identifier."""
        return "yolo_face_detection"
    
    def _make_anchors(self, feats_hw: List[Tuple[int, int]], grid_cell_offset: float = 0.5) -> Dict[int, np.ndarray]:
        """
        Generate anchors from features.
        
        Args:
            feats_hw: List of (height, width) for each feature level
            grid_cell_offset: Grid cell offset
            
        Returns:
            dict: Anchor points for each stride
        """
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points
    
    def _softmax(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Apply softmax function."""
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        return x_exp / x_sum
    
    def _resize_image(self, srcimg: np.ndarray, keep_ratio: bool = True) -> Tuple[np.ndarray, int, int, int, int]:
        """
        Resize image to model input size.
        
        Args:
            srcimg: Source image
            keep_ratio: Whether to keep aspect ratio
            
        Returns:
            Tuple of (resized_image, new_height, new_width, top_padding, left_padding)
        """
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
        
        return img, newh, neww, top, left
    
    def _detect_faces(self, srcimg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Internal face detection method.
        
        Args:
            srcimg: Source image
            
        Returns:
            Tuple of (bboxes, confidences, class_ids, landmarks)
        """
        # Preprocess image
        input_img, newh, neww, padh, padw = self._resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0
        
        # Create blob and run inference
        blob = cv2.dnn.blobFromImage(input_img)
        inputs = {self.session.get_inputs()[0].name: blob}
        outputs = self.session.run(None, inputs)
        
        # Post-process results
        det_bboxes, det_conf, det_classid, landmarks = self._post_process(
            outputs, scale_h, scale_w, padh, padw, srcimg.shape)
        
        return det_bboxes, det_conf, det_classid, landmarks
    
    def _post_process(self, preds: List[np.ndarray], scale_h: float, scale_w: float, 
                     padh: int, padw: int, orig_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Post-process model predictions.
        
        Args:
            preds: Model predictions
            scale_h: Height scaling factor
            scale_w: Width scaling factor
            padh: Height padding
            padw: Width padding
            orig_shape: Original image shape (height, width, channels)
            
        Returns:
            Tuple of (bboxes, scores, class_ids, landmarks)
        """
        bboxes, scores, landmarks = [], [], []
        
        for i, pred in enumerate(preds):
            stride = self.strides[i]
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))  # x1,y1,score1, ..., x5,y5,score5
            
            # Process bounding boxes
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self._softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))
            
            bbox = self._distance2bbox(self.anchors[stride], bbox_pred, 
                                     max_shape=(self.input_height, self.input_width)) * stride
            
            # Process keypoints
            if stride in self.anchors and len(kpts) > 0:
                anchor_pts = self.anchors[stride]
                if len(anchor_pts) >= len(kpts):
                    kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (anchor_pts[:len(kpts), 0].reshape((-1, 1)) - 0.5)) * stride
                    kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (anchor_pts[:len(kpts), 1].reshape((-1, 1)) - 0.5)) * stride
                    kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))
            
            # Adjust for padding and scaling
            bbox -= np.array([[padw, padh, padw, padh]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            
            # Clamp bounding boxes to valid ranges
            bbox[:, 0] = np.maximum(0, bbox[:, 0])  # x1 >= 0
            bbox[:, 1] = np.maximum(0, bbox[:, 1])  # y1 >= 0
            bbox[:, 2] = np.minimum(bbox[:, 2], orig_shape[1])  # x2 <= width
            bbox[:, 3] = np.minimum(bbox[:, 3], orig_shape[0])  # y2 <= height
            
            # Ensure x2 > x1 and y2 > y1
            bbox[:, 2] = np.maximum(bbox[:, 2], bbox[:, 0] + 1)
            bbox[:, 3] = np.maximum(bbox[:, 3], bbox[:, 1] + 1)
            
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))
            
            # Clamp landmarks to valid ranges
            for j in range(0, 15, 3):
                kpts[:, j] = np.maximum(0, np.minimum(kpts[:, j], orig_shape[1] - 1))  # x coordinate
                kpts[:, j+1] = np.maximum(0, np.minimum(kpts[:, j+1], orig_shape[0] - 1))  # y coordinate
            
            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)
        
        # Concatenate results
        if len(bboxes) == 0:
            return np.array([]), np.array([]), np.array([]), []
            
        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
        
        # Convert to xywh format
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]
        
        # Apply NMS
        if len(bboxes_wh) > 0:
            # Ensure all bboxes have positive width and height
            valid_mask = (bboxes_wh[:, 2] > 0) & (bboxes_wh[:, 3] > 0)
            if np.any(valid_mask):
                bboxes_wh = bboxes_wh[valid_mask]
                confidences = confidences[valid_mask]
                classIds = classIds[valid_mask]
                landmarks = landmarks[valid_mask]
            
            if len(bboxes_wh) > 0:
                indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
                if len(indices) > 0:
                    indices = indices.flatten()
                    return bboxes_wh[indices], confidences[indices], classIds[indices], landmarks[indices]
        
        # Return empty results if no detections
        return np.array([]), np.array([]), np.array([]), []
    
    def _distance2bbox(self, points: np.ndarray, distance: np.ndarray, 
                      max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert distance predictions to bounding boxes.
        
        Args:
            points: Anchor points
            distance: Distance predictions
            max_shape: Maximum shape for clipping
            
        Returns:
            np.ndarray: Bounding boxes
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def _initialize_boxmot(self):
        """
        Initialize BOXMOT tracker for face tracking.
        """
        if not BOXMOT_AVAILABLE:
            print("[ERROR] BOXMOT is not available. Cannot initialize BOXMOT tracker.")
            return
            
        try:
            # Get tracker configuration
            tracker_conf = get_tracker_config(self.boxmot_tracker_type)
            
            print(f"[INFO] Using default BOXMOT config: {tracker_conf}")

            reid_weights = Path(self.boxmot_reid_model) if self.boxmot_reid_model else None

            self.boxmot_tracker = create_tracker(
                tracker_type=self.boxmot_tracker_type,
                reid_weights=reid_weights,
                tracker_config=tracker_conf,
                device=self.device,
                half=False,  # Use float32 for compatibility
                per_class=True,  # Track all classes together
            )
            print(f"[INFO] BOXMOT face tracker '{self.boxmot_tracker_type}' initialized successfully")
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to initialize BOXMOT face tracker: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            print("[INFO] Falling back to simple enumeration")
            self.use_boxmot = False
    
    def _apply_boxmot_tracking(self, face_boxes: np.ndarray, face_scores: np.ndarray, face_classids: np.ndarray, image: np.ndarray) -> List[int]:
        """
        Apply BOXMOT tracking to face detections.
        
        Args:
            face_boxes: Detected face bounding boxes in (x, y, w, h) format
            face_scores: Detection confidence scores
            face_classids: Class IDs (should all be 0 for face class)
            image: Input image for tracking
            
        Returns:
            List of track IDs corresponding to each detection
        """
        try:
            # Convert face detections to BOXMOT format
            dets = self._create_boxmot_detections(face_boxes, face_scores, face_classids)
            
            if self.debug:
                print(f"[DEBUG BOXMOT] Created {len(dets)} detections for tracking")
                
            # Update BOXMOT tracker
            tracks = self.boxmot_tracker.update(dets, image)
            
            if self.debug:
                print(f"[DEBUG BOXMOT] Tracker returned {len(tracks) if tracks is not None else 0} tracks")
                
            # Extract track IDs
            track_ids = []
            if tracks is not None and len(tracks) > 0:
                for track in tracks:
                    # BOXMOT track format: [x1, y1, x2, y2, track_id, conf, class_id, det_ind]
                    if len(track) >= 8:
                        track_id = int(track[4])
                        det_ind = int(track[7])
                        track_ids.append(track_id)
                        
                        if self.debug:
                            print(f"[DEBUG BOXMOT] Track ID: {track_id}, Detection Index: {det_ind}")
                    else:
                        print(f"[WARNING] Invalid track format: {track}")
                        track_ids.append(len(track_ids))  # Fallback to enumeration
            else:
                # Fallback to simple enumeration if tracking fails
                track_ids = list(range(len(face_boxes)))
                if self.debug:
                    print("[DEBUG BOXMOT] No tracks returned, using simple enumeration")
                    
            return track_ids
            
        except Exception as e:
            print(f"[ERROR] BOXMOT tracking failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple enumeration
            return list(range(len(face_boxes)))
    
    def _create_boxmot_detections(self, face_boxes: np.ndarray, face_scores: np.ndarray, face_classids: np.ndarray) -> np.ndarray:
        """
        Convert face detection results to BOXMOT detection format.
        
        Args:
            face_boxes: Face bounding boxes in (x, y, w, h) format
            face_scores: Detection confidence scores
            face_classids: Class IDs
            
        Returns:
            numpy array of detections in BOXMOT format [x1, y1, x2, y2, conf, class_id]
        """
        dets = []
        
        for i in range(len(face_boxes)):
            x, y, w, h = face_boxes[i]
            conf = face_scores[i]
            class_id = face_classids[i]  # Should be 0 for faces
            
            # Convert from (x, y, w, h) to (x1, y1, x2, y2) format
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Create BOXMOT detection: [x1, y1, x2, y2, conf, class_id]
            det = np.array([x1, y1, x2, y2, conf, class_id])
            dets.append(det)
            
        return np.array(dets) if dets else np.empty((0, 6))
