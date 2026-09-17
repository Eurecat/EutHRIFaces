#!/usr/bin/env python3
"""
Decode compressed camera frames on the GPU when possible.

Several nodes decode the same 1080p JPEG stream on every frame. On a loaded Jetson that
was the largest CPU cost of those nodes (cv2.imdecode: ~8 ms alone, 50-80 ms under load).
nvJPEG through torchvision decodes on the GPU in ~2-3 ms including the copy back.
Falls back to cv2.imdecode when CUDA/torchvision are unavailable or a frame is not a JPEG
(e.g. PNG), so the result is always the same BGR uint8 array cv2 would return.
Set env JPEG_DECODER_DEVICE=cpu to force cv2.imdecode (e.g. to save GPU memory on small GPUs).

Note: identical copies of this file live in the other packages that decode camera frames
(face_recognition, visual_speech_activity, skeleton_detection, hand_gesture_recognition),
so no package depends on another one just for this helper.
"""
import os
from typing import Any, Optional

import cv2
import numpy as np

_MAX_GPU_FAILURES = 5


class JpegDecoder:
    def __init__(self, device: str = 'cuda', logger: Any = None):
        self._logger = logger
        self._gpu = False
        self.last_gpu_rgb = None  # uint8 3xHxW CUDA tensor of the last GPU-decoded frame (None after CPU decode)
        self._gpu_failures = 0
        device = os.environ.get('JPEG_DECODER_DEVICE', device)
        if not str(device).startswith('cuda'):
            return
        try:
            import torch
            from torchvision.io import ImageReadMode, decode_jpeg
            if not torch.cuda.is_available():
                self._log('info', 'JpegDecoder: CUDA not available, using cv2.imdecode')
                return
            self._torch = torch
            self._decode_jpeg = decode_jpeg
            self._rgb_mode = ImageReadMode.RGB
            self._device = device
            self._gpu = True
            self._log('info', f'JpegDecoder: decoding frames on GPU (nvJPEG, {device})')
        except Exception as e:  # torchvision missing or built without nvJPEG
            self._log('info', f'JpegDecoder: GPU decoding unavailable ({e}), using cv2.imdecode')

    @property
    def uses_gpu(self) -> bool:
        return self._gpu

    def decode_bgr(self, data) -> Optional[np.ndarray]:
        """Decode a CompressedImage payload into a BGR uint8 HxWx3 array (None on failure)."""
        if self._gpu:
            try:
                encoded = self._torch.frombuffer(data, dtype=self._torch.uint8)
                rgb = self._decode_jpeg(encoded, mode=self._rgb_mode, device=self._device)
                self.last_gpu_rgb = rgb
                return rgb.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
            except Exception as e:
                self._gpu_failures += 1
                if self._gpu_failures >= _MAX_GPU_FAILURES:
                    self._gpu = False
                    self._log('warn', f'JpegDecoder: GPU decoding failed {self._gpu_failures} times ({e}), '
                                      'switching to cv2.imdecode')
        self.last_gpu_rgb = None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def _log(self, level: str, message: str) -> None:
        if self._logger is not None:
            getattr(self._logger, level)(message)
