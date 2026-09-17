"""Per-face quality score used to decide whether an embedding may create or teach an identity.

Measured on video_3.mp4 (5 people, 2289 detections):
  - the inter-ocular distance divided by the face-box width drops with head yaw. Between
    0.06 and 0.14 (3/4 view turning into profile) the embedding picks the right person
    only 73-96% of the time with negative best-vs-second margins; above 0.14 it is 97-99%.
  - hand false positives have bbox confidence <= 0.25; real faces are >= 0.53 (p5).

The score is 0 for faces that must never create or teach an identity and grows to 1
for near-frontal, confidently detected faces.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

# FacialLandmarks indices always filled from the YOLO 5-point detector
RIGHT_EYE_INSIDE = 39
LEFT_EYE_INSIDE = 42


@dataclass
class FaceQualityConfig:
    min_detection_confidence: float = 0.40
    profile_eye_ratio: float = 0.10   # eye distance / face width at or below -> quality 0
    frontal_eye_ratio: float = 0.18   # at or above -> full yaw score


def eye_ratio(bbox_xyxy: Sequence[float], left_eye_x: Optional[float], right_eye_x: Optional[float]) -> Optional[float]:
    """Horizontal inter-ocular distance over face-box width (all normalized coordinates)."""
    if left_eye_x is None or right_eye_x is None:
        return None
    width = float(bbox_xyxy[2]) - float(bbox_xyxy[0])
    if width <= 1e-6:
        return None
    return abs(float(right_eye_x) - float(left_eye_x)) / width


def face_quality(bbox_xyxy: Sequence[float], bbox_confidence: float,
                 left_eye_x: Optional[float], right_eye_x: Optional[float],
                 config: FaceQualityConfig = FaceQualityConfig()) -> float:
    if bbox_confidence < config.min_detection_confidence:
        return 0.0
    ratio = eye_ratio(bbox_xyxy, left_eye_x, right_eye_x)
    if ratio is None:
        return 0.0
    span = max(1e-6, config.frontal_eye_ratio - config.profile_eye_ratio)
    return float(min(1.0, max(0.0, (ratio - config.profile_eye_ratio) / span)))


def face_quality_from_msg(msg, config: FaceQualityConfig = FaceQualityConfig()) -> float:
    """Quality of an hri_msgs/FacialLandmarks message."""
    box = msg.bbox_xyxy
    left = right = None
    if len(msg.landmarks) > max(RIGHT_EYE_INSIDE, LEFT_EYE_INSIDE):
        right_eye = msg.landmarks[RIGHT_EYE_INSIDE]
        left_eye = msg.landmarks[LEFT_EYE_INSIDE]
        if right_eye.c > 0 and left_eye.c > 0:
            left, right = left_eye.x, right_eye.x
    return face_quality((box.xmin, box.ymin, box.xmax, box.ymax), float(msg.bbox_confidence), left, right, config)
