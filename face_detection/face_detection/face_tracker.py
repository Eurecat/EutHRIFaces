#!/usr/bin/env python3
"""
Lightweight dependency-free face tracker.

Gives each detected face a track id that stays the same from frame to frame, so
``face_<id>`` keeps pointing at the same person. Downstream nodes (face
recognition, visual speech activity, person manager) key their per-face state on
that id, so an id that jumps between people makes identities swap on screen.

Faces rarely overlap, so greedy matching on IoU with a center-distance fallback
(for fast motion at low frame rate) is enough.
"""
from typing import Dict, List, Sequence


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class IouFaceTracker:
    """Greedy IoU tracker with center-distance fallback.

    Args:
        iou_threshold: minimum IoU to match a detection to a track directly.
        max_center_distance: fallback gate when IoU is low, as a fraction of the
            face size (sqrt of the larger box area).
        max_size_ratio: fallback matches must not change box area by more than this factor.
        max_missed_frames: frames a track survives without a matching detection.
    """

    def __init__(self, iou_threshold: float = 0.2, max_center_distance: float = 1.0,
                 max_size_ratio: float = 2.5, max_missed_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_size_ratio = max_size_ratio
        self.max_missed_frames = max_missed_frames
        self._tracks: Dict[int, Dict] = {}  # id -> {'box': [x1,y1,x2,y2], 'missed': int}
        self._next_id = 0

    def reset(self):
        self._tracks.clear()

    def update(self, boxes_xyxy: Sequence[Sequence[float]]) -> List[int]:
        """Match this frame's boxes to tracks. Returns one track id per box, in input order."""
        boxes = [[float(v) for v in b[:4]] for b in boxes_xyxy]
        candidates = []  # (matched_by_iou, score, track_id, det_index)
        for tid, track in self._tracks.items():
            tb = track['box']
            t_area = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1]))
            tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            for di, db in enumerate(boxes):
                iou = _iou(tb, db)
                if iou >= self.iou_threshold:
                    candidates.append((1, iou, tid, di))
                    continue
                d_area = max(1e-6, (db[2] - db[0]) * (db[3] - db[1]))
                ratio = max(t_area, d_area) / min(t_area, d_area)
                if ratio > self.max_size_ratio:
                    continue
                size = max(t_area, d_area) ** 0.5
                dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5 / size
                if dist <= self.max_center_distance:
                    candidates.append((0, -dist, tid, di))

        # IoU matches first (highest IoU), then distance matches (closest)
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        ids: List[int] = [-1] * len(boxes)
        used_tracks = set()
        for _, _, tid, di in candidates:
            if tid in used_tracks or ids[di] != -1:
                continue
            ids[di] = tid
            used_tracks.add(tid)

        for tid in list(self._tracks):
            if tid in used_tracks:
                self._tracks[tid]['missed'] = 0
            else:
                self._tracks[tid]['missed'] += 1
                if self._tracks[tid]['missed'] > self.max_missed_frames:
                    del self._tracks[tid]

        for di, box in enumerate(boxes):
            if ids[di] == -1:
                ids[di] = self._next_id
                self._next_id += 1
            self._tracks[ids[di]] = {'box': box, 'missed': 0}
        return ids
