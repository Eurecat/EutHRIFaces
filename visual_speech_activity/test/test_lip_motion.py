"""Unit tests for the landmark lip-motion speaking detector (no ROS needed)."""
import importlib.util
import math
from pathlib import Path
from unittest import mock

_spec = importlib.util.spec_from_file_location(
    "lip_motion_detector",
    Path(__file__).resolve().parents[1] / "visual_speech_activity" / "lip_motion_detector.py")
lip_motion_detector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lip_motion_detector)
LipMotionDetector = lip_motion_detector.LipMotionDetector


def _landmarks(opening):
    """68 landmarks with a 40 px wide inner mouth opened by `opening` px."""
    lms = [(0.0, 0.0, 1.0)] * 68
    lms[60], lms[64] = (80.0, 100.0, 1.0), (120.0, 100.0, 1.0)
    lms[62], lms[66] = (100.0, 100.0 - opening / 2, 1.0), (100.0, 100.0 + opening / 2, 1.0)
    return lms


def _feed(detector, openings, face_id="face_0"):
    result = None
    for k, opening in enumerate(openings):
        with mock.patch.object(lip_motion_detector.time, "monotonic", return_value=k * 0.1):
            result = detector.detect_speaking(None, _landmarks(opening), None, face_id)
    return result


def test_talking_mouth_is_speaking_and_still_open_mouth_is_not():
    detector = LipMotionDetector()
    talking = [8 + 8 * math.sin(k * 1.9) for k in range(10)]  # opening varies 0..16 px
    speaking, confidence, bbox = _feed(detector, talking, "talker")
    assert speaking and confidence > 0.5 and bbox is not None
    speaking, confidence, _ = _feed(detector, [12.0] * 10, "yawn")  # open but not moving
    assert not speaking and confidence == 0.0


def test_yolo_only_landmarks_report_nothing():
    lms = [(0.0, 0.0, 0.0)] * 70
    assert LipMotionDetector().detect_speaking(None, lms, None, "f") == (False, 0.0, None)


def test_needs_min_samples_and_forgets_old_faces():
    detector = LipMotionDetector(min_samples=5)
    speaking, confidence, _ = _feed(detector, [0, 16, 0], "f")
    assert (speaking, confidence) == (False, 0.0)
    with mock.patch.object(lip_motion_detector.time, "monotonic", return_value=100.0):
        detector.cleanup_old_identities([])
    assert "f" not in detector._history
