"""Evaluator maths and candidate methods on synthetic landmarks (no ROS needed)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import evaluate_vsad as ev  # noqa: E402
from vsad_methods import METHODS  # noqa: E402


def face(opening, width=40.0, noise=0.0, rng=None):
    lms = np.zeros((68, 3), np.float32)
    lms[:, 2] = 1.0
    lms[48], lms[54] = (80, 100, 1), (120, 100, 1)
    lms[60], lms[64] = (80, 100, 1), (120, 100, 1)
    for top, bot in ((61, 67), (62, 66), (63, 65), (50, 58), (51, 57), (52, 56)):
        lms[top, :2] = (100, 100 - opening / 2)
        lms[bot, :2] = (100, 100 + opening / 2)
    if noise:
        lms[:, :2] += rng.normal(0, noise, (68, 2))
    return lms


def stream(label_fn, rate, seconds=20, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(int(seconds * rate)):
        t = k / rate
        frames.append({"t": t, "bag_t": t, "face": "f", "image": None, "bbox": (60, 40, 140, 160),
                       "lms": face(label_fn(t), noise=noise, rng=rng)})
    return frames


def speak_block(t):          # speaks in the first 10 s, still mouth after
    return 6 + 6 * math.sin(t * 17.0) if t < 10 else 3.0


def labels():
    return [(0.0, "SPEAK"), (10.0, "SILENT")]


def test_auc_and_f1_on_separable_scores():
    assert ev.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert ev.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert ev.best_f1([0.9, 0.8, 0.2, 0.1] * 10, [1, 1, 0, 0] * 10)[0] == 1.0


def test_decimate_keeps_target_rate():
    assert abs(len(ev.decimate(stream(speak_block, 15), 5)) - 100) <= 3


def test_every_landmark_method_separates_talking_from_still_at_both_rates():
    for rate in (15, 5):
        frames = ev.decimate(stream(speak_block, 15, noise=0.3), rate)
        for name, cls in METHODS.items():
            if name == "crop_diff":
                continue
            rows = ev.run_method(cls, frames, labels(), [])
            assert rows, name
            assert ev.auc([r[1] for r in rows], [r[2] for r in rows]) > 0.9, (name, rate)


def test_corrections_override_and_main_face(tmp_path):
    labels = [(0.0, "SPEAK"), (10.0, "SILENT"), (20.0, "SPEAK")]
    (tmp_path / "c.json").write_text('[{"from_s": 12, "to_s": 15, "label": "TRANSITION"}]')
    fixed = ev.apply_corrections(labels, str(tmp_path / "c.json"))
    assert ev.lookup(fixed, 11) == "SILENT" and ev.lookup(fixed, 13) == "TRANSITION" and ev.lookup(fixed, 16) == "SILENT"
    mk = lambda face, t: {"face": face, "bag_t": t, "t": t}
    frames = [mk("a", 1)] * 5 + [mk("b", 2)] * 2
    assert {f["face"] for f in ev.main_face_only(frames, [(0.0, "dist=1m")])} == {"a"}
