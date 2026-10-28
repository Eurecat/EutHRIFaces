#!/usr/bin/env python3
"""
Compare lip speaking methods on a ground-truth bag (see record_vsad_session.py).

For every method and every simulated frame rate (the bag is decimated, so one 15 Hz recording
also answers "what at 5 Hz?") it prints, over the frames with a SPEAK / SILENT label:
AUC (threshold-free quality), best F1 and the threshold giving it, and the same split by scenario
block (distance), by face size in pixels, and for the hard negatives (smile, chew, yawn, head turn).

inside the eut_visual_speech_activity container (needs rosbag2_py + hri_msgs):

    python3 evaluate_vsad.py /workspace/log/vsad/session_a --rates 15 10 5 --cache
    python3 evaluate_vsad.py /workspace/log/vsad/session_a --methods mar3_std crop_diff

--cache writes <bag>/vsad_frames.pkl on the first run; later runs (and other machines with only
numpy) read it instead of the bag. --images adds the camera frames (needed by crop_diff; slow).
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vsad_methods import METHODS  # noqa: E402

LABEL_TOPIC, PHASE_TOPIC = "/gt/speech_label", "/gt/phase"
LANDMARK_TOPIC, IMAGE_TOPIC = "/humans/faces/detected", "/camera/image_raw/compressed"
SIZE_BUCKETS = [(0, 60), (60, 120), (120, 240), (240, 10_000)]   # face bbox height, px


def read_bag(path, with_images):
    """-> frames: list of dict(t, face, lms (68,3), bbox px, image|None); labels, phases: [(t, str)]."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import cv2

    storage = "mcap" if any(f.endswith(".mcap") for f in os.listdir(path)) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {LABEL_TOPIC, PHASE_TOPIC, LANDMARK_TOPIC} | ({IMAGE_TOPIC} if with_images else set())
    classes = {n: get_message(types[n]) for n in wanted if n in types}
    for need in (LABEL_TOPIC, LANDMARK_TOPIC):
        if need not in classes:
            sys.exit(f"{path}: topic {need} not in bag (has {sorted(types)})")

    frames, labels, phases, images = [], [], [], []
    while reader.has_next():
        name, raw, bag_t = reader.read_next()
        if name not in classes:
            continue
        msg = deserialize_message(raw, classes[name])
        if name == LABEL_TOPIC:
            labels.append((bag_t * 1e-9, msg.data))
        elif name == PHASE_TOPIC:
            phases.append((bag_t * 1e-9, msg.data))
        elif name == IMAGE_TOPIC:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            images.append((stamp, cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)))
        else:
            for face in msg.ids:
                if len(face.landmarks) < 68:
                    continue
                w, h = face.width, face.height
                lms = np.array([(p.x * w, p.y * h, p.c) for p in face.landmarks], dtype=np.float32)
                b = face.bbox_xyxy
                stamp = face.header.stamp.sec + face.header.stamp.nanosec * 1e-9
                frames.append({"t": stamp, "bag_t": bag_t * 1e-9, "face": face.face_id, "lms": lms,
                               "bbox": (b.xmin * w, b.ymin * h, b.xmax * w, b.ymax * h), "image": None})
    if images:
        stamps = np.array([s for s, _ in images])
        for fr in frames:
            fr["image"] = images[int(np.argmin(np.abs(stamps - fr["t"])))][1]
    return frames, labels, phases


def lookup(events, t):
    """Value of the last event at or before t (events sorted by time)."""
    value = None
    for et, ev in events:
        if et > t:
            break
        value = ev
    return value


def apply_corrections(labels, path):
    """corrections.json next to the bag: [{"from_s": 231, "to_s": 237, "label": "TRANSITION", "why": "..."}],
    seconds counted from the first ground-truth label. Used when the person slipped (spoke in a
    SILENT block): the bag stays untouched, the label is overridden when scoring."""
    if not os.path.exists(path):
        return labels
    import json
    fixes = json.load(open(path))
    t0 = labels[0][0]
    out = []
    for i, (t, label) in enumerate(labels):
        for fix in fixes:
            if t0 + fix["from_s"] <= t < t0 + fix["to_s"]:
                label = fix["label"]
        out.append((t, label))
    # a block starts at the previous label message: carry a fix that starts between two messages
    for fix in fixes:
        start = t0 + fix["from_s"]
        if not any(abs(t - start) < 1e-3 for t, _ in out):
            out.append((start, fix["label"]))
            end = t0 + fix["to_s"]
            before = [l for t, l in labels if t <= end]
            out.append((end, before[-1] if before else "TRANSITION"))
    return sorted(out)


def main_face_only(frames, phases):
    """Keep, per distance block, only the face id seen most often there (the person recorded, not
    a colleague walking in)."""
    counts = {}
    for fr in frames:
        key = (lookup(phases, fr["bag_t"]) or "?").split(",")[0]
        counts.setdefault(key, {}).setdefault(fr["face"], 0)
        counts[key][fr["face"]] += 1
    main = {k: max(v, key=v.get) for k, v in counts.items()}
    return [fr for fr in frames if main[(lookup(phases, fr["bag_t"]) or "?").split(",")[0]] == fr["face"]]


def decimate(frames, rate):
    """Keep about `rate` frames per second per face."""
    if not rate:
        return frames
    last, out = {}, []
    for fr in frames:
        if fr["t"] - last.get(fr["face"], -1e9) >= 0.92 / rate:
            last[fr["face"]] = fr["t"]
            out.append(fr)
    return out


def auc(scores, truth):
    """Probability a random SPEAK frame outscores a random SILENT frame (rank-based)."""
    s, y = np.asarray(scores), np.asarray(truth)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_f1(scores, truth):
    s, y = np.asarray(scores), np.asarray(truth)
    best = (0.0, float("nan"))
    for thr in np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97))):
        pred = s >= thr
        tp = int((pred & (y == 1)).sum())
        fp, fn = int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


def run_method(cls, frames, labels, phases, with_labels_only=True):
    """-> rows (t, score, truth, phase, face_h)."""
    state, rows = {}, []
    for fr in frames:
        det = state.setdefault(fr["face"], cls())
        score = det.update(fr["t"], fr["lms"], fr["bbox"], fr["image"])
        label = lookup(labels, fr["bag_t"])
        if score is None or label not in ("SPEAK", "SILENT"):
            continue
        rows.append((fr["t"], score, 1 if label == "SPEAK" else 0,
                     lookup(phases, fr["bag_t"]) or "?", fr["bbox"][3] - fr["bbox"][1]))
    return rows


def report(name, rate, rows, thr_global=None):
    if not rows:
        print(f"  {name:22s} no scored frames")
        return None
    score = [r[1] for r in rows]
    truth = [r[2] for r in rows]
    f1, thr = best_f1(score, truth)
    print(f"  {name:22s} AUC {auc(score, truth):.3f}  bestF1 {f1:.3f} @thr {thr:.4f}  "
          f"({sum(truth)} speak / {len(truth) - sum(truth)} silent frames)")
    return thr


def breakdown(name, rows, thr):
    """AUC per phase / face size, and the false-positive rate at the global threshold."""
    groups = {}
    for _, s, y, phase, h in rows:
        groups.setdefault(("phase", phase.split(",")[0]), []).append((s, y))
        for lo, hi in SIZE_BUCKETS:
            if lo <= h < hi:
                groups.setdefault(("face_px", f"{lo}-{hi}"), []).append((s, y))
        if phase.endswith(",hard"):
            groups.setdefault(("hard_neg", "all"), []).append((s, y))
    parts = []
    for (kind, key), vals in sorted(groups.items()):
        s, y = [v[0] for v in vals], [v[1] for v in vals]
        if kind == "hard_neg":
            parts.append(f"hard-neg false-pos {np.mean(np.asarray(s) >= thr):.0%}")
        elif 0 < sum(y) < len(y):
            parts.append(f"{kind} {key}: AUC {auc(s, y):.2f}")
    print(f"      {name}: " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--rates", type=float, nargs="+", default=[0, 10, 5], help="0 = native rate")
    ap.add_argument("--methods", nargs="+", default=None, help="subset (names from vsad_methods.METHODS)")
    ap.add_argument("--images", action="store_true", help="read camera frames (crop_diff)")
    ap.add_argument("--cache", action="store_true", help="read/write <bag>/vsad_frames.pkl")
    ap.add_argument("--all-faces", action="store_true", help="do not reduce to the main face per distance block")
    ap.add_argument("--detail", action="store_true", help="also print the per-distance / face size split")
    args = ap.parse_args()

    cache = os.path.join(args.bag, "vsad_frames.pkl")
    if args.cache and os.path.exists(cache):
        frames, labels, phases = pickle.load(open(cache, "rb"))
        print(f"read {cache}")
    else:
        frames, labels, phases = read_bag(args.bag, args.images)
        if args.cache:
            pickle.dump((frames, labels, phases), open(cache, "wb"))
    if not frames or not labels:
        sys.exit("no landmark frames or no ground-truth labels in the bag")
    labels = apply_corrections(labels, os.path.join(args.bag, "corrections.json"))
    if not args.all_faces:
        n = len(frames)
        frames = main_face_only(frames, phases)
        print(f"main face per distance block: kept {len(frames)} of {n} face frames")
    span = frames[-1]["t"] - frames[0]["t"]
    print(f"{len(frames)} face frames over {span:.0f} s ({len(frames) / max(span, 1e-6):.1f} Hz total), "
          f"{sum(1 for _, l in labels if l == 'SPEAK')} SPEAK labels")

    names = args.methods or [n for n in METHODS if args.images or n != "crop_diff"]
    for rate in args.rates:
        print(f"\n=== {'native rate' if not rate else f'{rate:g} Hz'} ===")
        sub = decimate(frames, rate)
        for name in names:
            rows = run_method(METHODS[name], sub, labels, phases)
            thr = report(name, rate, rows)
            if args.detail and rows:
                breakdown(name, rows, thr)


if __name__ == "__main__":
    main()
