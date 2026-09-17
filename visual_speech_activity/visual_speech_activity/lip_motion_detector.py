#!/usr/bin/env python3
"""
Visual speech activity from lip motion in the 68-point landmarks.

Speaking shows up as the inner-lip opening changing over time, not as the mouth being
open (a smile or a resting open mouth is not speech). The score is the standard deviation
of the mouth aspect ratio (inner lip gap / inner mouth width) over a short time window.

Measured on video_3 with MediaPipe and face-mesh ONNX landmarks: the mouth aspect ratio
agrees between both landmark sources (correlation 0.79-0.92) and is smooth frame to frame
(autocorrelation 0.77-0.94), while VSDLM's per-frame output on the same faces jumped
between frames (autocorrelation 0.07-0.70) because it depends on the exact mouth crop.

Same interface as VSDLMDetector.detect_speaking, so the node can use either one.
"""
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# dlib 68-point layout (ros4hri FacialLandmarks)
_INNER_LEFT, _INNER_RIGHT = 60, 64
_INNER_TOP, _INNER_BOTTOM = 62, 66
_MOUTH = list(range(48, 68))


class LipMotionDetector:
    """Speaking confidence from the variation of the lip opening.

    Args:
        window_s: Time window of mouth-aspect-ratio samples.
        min_samples: Samples needed in the window before reporting any confidence.
        min_std: Standard deviation at or below which confidence is 0 (landmark jitter).
        full_std: Standard deviation at or above which confidence is 1.
        speaking_threshold: Confidence at or above which the face is reported as speaking.
        track_timeout_s: Forget a face not seen for this long.
    """

    def __init__(self, window_s: float = 1.0, min_samples: int = 5, min_std: float = 0.02,
                 full_std: float = 0.08, speaking_threshold: float = 0.5, track_timeout_s: float = 2.0,
                 logger=None):
        self.window_s = window_s
        self.min_samples = min_samples
        self.min_std = min_std
        self.full_std = max(full_std, min_std + 1e-6)
        self.speaking_threshold = speaking_threshold
        self.track_timeout_s = track_timeout_s
        self.logger = logger
        self._history: Dict[str, Deque[Tuple[float, float]]] = {}

    @staticmethod
    def mouth_aspect_ratio(landmarks: List[Tuple[float, ...]]) -> Optional[float]:
        if len(landmarks) < 68:
            return None
        needed = (_INNER_LEFT, _INNER_RIGHT, _INNER_TOP, _INNER_BOTTOM)
        if any(len(landmarks[i]) >= 3 and landmarks[i][2] <= 0 for i in needed):
            return None  # YOLO 5-point faces carry no lip contour
        point = lambda i: np.array(landmarks[i][:2], dtype=np.float64)
        width = np.linalg.norm(point(_INNER_RIGHT) - point(_INNER_LEFT))
        if width < 1e-6:
            return None
        return float(np.linalg.norm(point(_INNER_BOTTOM) - point(_INNER_TOP)) / width)

    def detect_speaking(self, image, landmarks: List[Tuple[float, ...]], face_bbox=None,
                        face_id: Optional[str] = None) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """(is_speaking, confidence, mouth bbox in pixels). The image is not used."""
        mar = self.mouth_aspect_ratio(landmarks)
        if mar is None:
            return False, 0.0, None
        now = time.monotonic()
        key = face_id or "_single"
        history = self._history.setdefault(key, deque())
        history.append((now, mar))
        while history and now - history[0][0] > self.window_s:
            history.popleft()

        xs = [landmarks[i][0] for i in _MOUTH]
        ys = [landmarks[i][1] for i in _MOUTH]
        mouth_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        if len(history) < self.min_samples:
            return False, 0.0, mouth_bbox
        std = float(np.std([m for _, m in history]))
        confidence = float(np.clip((std - self.min_std) / (self.full_std - self.min_std), 0.0, 1.0))
        return confidence >= self.speaking_threshold, confidence, mouth_bbox

    def reset_identity(self, face_id: str):
        self._history.pop(face_id, None)

    def cleanup_old_identities(self, active_face_ids: List[str]):
        active = set(active_face_ids)
        now = time.monotonic()
        for key in [k for k, h in self._history.items()
                    if k not in active and (not h or now - h[-1][0] > self.track_timeout_s)]:
            del self._history[key]
