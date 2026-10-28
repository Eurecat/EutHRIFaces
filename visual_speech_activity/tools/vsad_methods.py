"""
Candidate speaking scores from lip landmarks (and optionally the frame), for offline comparison.

Every method has `update(t, lms, bbox, image) -> Optional[float]`: t is the frame time in seconds,
lms the 68 landmarks as an (68, 3) array (x, y in pixels, confidence), bbox (x0, y0, x1, y1) in
pixels, image the BGR frame or None. The score is higher for more speech, on an arbitrary scale
(the evaluator sweeps the threshold). None means "not enough data yet". One instance per face.

Time windows are in seconds, never in samples, so a method behaves the same at 5 and 15 Hz.
"""
from collections import deque
from typing import Dict, Optional

import numpy as np

INNER_PAIRS = ((61, 67), (62, 66), (63, 65))
OUTER_PAIRS = ((50, 58), (51, 57), (52, 56))


def _dist(lms, a, b):
    return float(np.hypot(*(lms[a, :2] - lms[b, :2])))


def mar_single(lms, bbox=None):
    w = _dist(lms, 60, 64)
    return _dist(lms, 62, 66) / w if w > 1e-6 else None


def mar_triple(lms, bbox=None):
    """Three inner-lip gaps averaged: landmark noise drops by about sqrt(3)."""
    w = _dist(lms, 60, 64)
    return sum(_dist(lms, a, b) for a, b in INNER_PAIRS) / 3 / w if w > 1e-6 else None


def outer_triple(lms, bbox=None):
    w = _dist(lms, 48, 54)
    return sum(_dist(lms, a, b) for a, b in OUTER_PAIRS) / 3 / w if w > 1e-6 else None


def gap_over_face(lms, bbox):
    """Lip gaps over the face height instead of the mouth width (the width is noisy when small)."""
    h = bbox[3] - bbox[1]
    if h < 1e-6:
        return None
    gaps = [_dist(lms, a, b) for a, b in INNER_PAIRS + OUTER_PAIRS]
    return sum(gaps) / len(gaps) / h * 4.0


class _Windowed:
    """Keeps (t, value) samples of the last `window_s` seconds."""

    window_s = 1.0
    min_span_frac = 0.5   # window part that must be covered before any score
    min_samples = 4
    feature = staticmethod(mar_single)

    def __init__(self, window_s=None):
        if window_s:
            self.window_s = window_s
        self.hist = deque()

    def _push(self, t, value):
        if not self.hist or self.hist[-1][0] != t:
            self.hist.append((t, value))
        while self.hist and t - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        return (len(self.hist) >= self.min_samples
                and self.hist[-1][0] - self.hist[0][0] >= self.min_span_frac * self.window_s)

    def update(self, t, lms, bbox, image=None):
        value = type(self).feature(lms, bbox)
        if value is None or not self._push(t, value):
            return None
        return self.score(np.array([v for _, v in self.hist]), np.array([x for x, _ in self.hist]))

    def score(self, values, times):
        raise NotImplementedError


class MarStd(_Windowed):
    """Current production method: std of the single inner-lip gap / mouth width."""
    def score(self, v, t):
        return float(np.std(v))


class Mar3Std(MarStd):
    feature = staticmethod(mar_triple)


class Outer3Std(MarStd):
    feature = staticmethod(outer_triple)


class GapFaceStd(MarStd):
    feature = staticmethod(gap_over_face)


class Mar3Range(_Windowed):
    """Spread between the 10th and 90th percentile: one outlier landmark frame does not count."""
    feature = staticmethod(mar_triple)

    def score(self, v, t):
        return float(np.percentile(v, 90) - np.percentile(v, 10))


class Mar3Diff(_Windowed):
    """Mean absolute frame-to-frame change per sample (blind to slow drift, e.g. a head tilt)."""
    feature = staticmethod(mar_triple)

    def score(self, v, t):
        return float(np.mean(np.abs(np.diff(v))))


class Mar3Detrended(_Windowed):
    """Std after removing a straight line: a mouth slowly opening is not speech."""
    feature = staticmethod(mar_triple)

    def score(self, v, t):
        if np.ptp(t) < 1e-6:
            return None
        resid = v - np.polyval(np.polyfit(t - t[0], v, 1), t - t[0])
        return float(np.std(resid))


class Combo(_Windowed):
    """Triple gap std relative to the face's own quiet level (adaptive floor, below)."""
    feature = staticmethod(mar_triple)

    def __init__(self, window_s=None):
        super().__init__(window_s)
        self.floor = None

    def score(self, v, t):
        s = float(np.std(v))
        # the floor follows the quietest window seen (slowly rising), so landmark noise of a small
        # face is subtracted instead of read as speech
        self.floor = s if self.floor is None else min(s, self.floor * 1.002 + 1e-6)
        return s - 0.9 * self.floor


class CropDiff:
    """Pixels: mean absolute change of the mouth crop between frames, averaged over the window.
    Needs the frame; independent of landmark noise but sees head motion too."""

    window_s = 1.0
    min_samples = 4

    def __init__(self, window_s=None):
        if window_s:
            self.window_s = window_s
        self.hist = deque()
        self.prev = None

    def update(self, t, lms, bbox, image=None):
        if image is None:
            return None
        import cv2
        mouth = lms[48:68, :2]
        x0, y0 = mouth.min(axis=0)
        x1, y1 = mouth.max(axis=0)
        w, h = x1 - x0, y1 - y0
        if w < 2:
            return None
        pad = 0.25 * w
        xa, xb = int(max(0, x0 - pad)), int(min(image.shape[1], x1 + pad))
        ya, yb = int(max(0, y0 - pad)), int(min(image.shape[0], y1 + pad))
        if xb - xa < 3 or yb - ya < 3:
            return None
        crop = cv2.cvtColor(image[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        crop = (crop - crop.mean()) / (crop.std() + 1e-3)   # lighting-invariant
        change = None if self.prev is None else float(np.mean(np.abs(crop - self.prev)))
        self.prev = crop
        if change is None:
            return None
        self.hist.append((t, change))
        while self.hist and t - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        if len(self.hist) < self.min_samples or self.hist[-1][0] - self.hist[0][0] < 0.5 * self.window_s:
            return None
        return float(np.mean([c for _, c in self.hist]))


METHODS: Dict[str, type] = {
    "mar_std (current)": MarStd,
    "mar3_std": Mar3Std,
    "outer3_std": Outer3Std,
    "gap_face_std": GapFaceStd,
    "mar3_range": Mar3Range,
    "mar3_diff": Mar3Diff,
    "mar3_detrended": Mar3Detrended,
    "mar3_adaptive": Combo,
    "crop_diff": CropDiff,
}
