"""5-point similarity alignment of face crops.

Measured on video_3.mp4 (docs/identity_experiments.md, entry 3): aligning FaceNet crops to
the standard 5-point template instead of cropping the raw detection box raised the
margin p10 between a face's own person and the best other person from 0.27 to 0.37.
"""

from typing import Optional

import cv2
import numpy as np

# Standard 112x112 5-point template (eyes, nose tip, mouth corners), image-left first
TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

# FacialLandmarks indices filled from the YOLO 5-point detector
EYE_INDICES = (42, 39)
NOSE_INDEX = 30
MOUTH_INDICES = (54, 48)


def five_points_from_msg(msg) -> Optional[np.ndarray]:
    """Pixel coordinates [left eye, right eye, nose, left mouth, right mouth] (image-left first)."""
    lms = msg.landmarks
    needed = EYE_INDICES + (NOSE_INDEX,) + MOUTH_INDICES
    if len(lms) <= max(needed) or any(lms[i].c <= 0 for i in needed):
        return None
    w, h = float(msg.width), float(msg.height)
    point = lambda i: (lms[i].x * w, lms[i].y * h)
    eyes = sorted(point(i) for i in EYE_INDICES)
    mouth = sorted(point(i) for i in MOUTH_INDICES)
    return np.array([eyes[0], eyes[1], point(NOSE_INDEX), mouth[0], mouth[1]], dtype=np.float32)


def align_face(image: np.ndarray, points: np.ndarray, size: int = 160) -> Optional[np.ndarray]:
    """Warp the face so its 5 points land on the template scaled to ``size`` x ``size``."""
    matrix, _ = cv2.estimateAffinePartial2D(points, TEMPLATE_112 * (size / 112.0), method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
