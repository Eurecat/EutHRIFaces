"""Unit tests for the built-in IoU face tracker (no ROS needed)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "face_tracker", Path(__file__).resolve().parents[1] / "face_detection" / "face_tracker.py")
face_tracker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(face_tracker)
IouFaceTracker = face_tracker.IouFaceTracker


def _box(cx, cy, size=60):
    return [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]


def test_ids_follow_faces_when_detection_order_changes():
    tracker = IouFaceTracker()
    a, b = tracker.update([_box(100, 100), _box(300, 100)])
    # Detector returns faces in a different order (e.g. sorted by score)
    ids = tracker.update([_box(305, 102), _box(98, 101)])
    assert ids == [b, a]


def test_close_faces_keep_their_ids_while_moving():
    tracker = IouFaceTracker()
    a, b = tracker.update([_box(200, 100), _box(270, 100)])
    for step in range(1, 20):
        left, right = _box(200 + step * 2, 100), _box(270 - step, 100)
        ids = tracker.update([right, left] if step % 2 else [left, right])
        assert ids == ([b, a] if step % 2 else [a, b])


def test_fast_motion_matches_by_center_distance():
    tracker = IouFaceTracker()
    (a,) = tracker.update([_box(100, 100)])
    # Moved 45 px (IoU below threshold) but still within one face size
    assert tracker.update([_box(145, 100)]) == [a]


def test_ids_are_never_reused_and_tracks_expire():
    tracker = IouFaceTracker(max_missed_frames=2)
    (a,) = tracker.update([_box(100, 100)])
    for _ in range(2):
        assert tracker.update([]) == []
    assert tracker.update([_box(100, 100)]) == [a]  # survived 2 missed frames
    for _ in range(3):
        tracker.update([])
    (c,) = tracker.update([_box(100, 100)])
    assert c != a


def test_far_face_gets_new_id():
    tracker = IouFaceTracker()
    (a,) = tracker.update([_box(100, 100)])
    (b,) = tracker.update([_box(400, 100)])
    assert b != a
