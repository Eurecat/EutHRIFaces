#!/usr/bin/env python3
"""
Build per-detection person labels for a dataset recorded on a looping test video.

Detections of the same video frame are identical across loops, so faces are linked
along the video timeline (frame index), not along tracker ids (which swap at every
loop wrap). Tracklets that cover most of the video are people; short ones are false
positives (label -1). A contact sheet is written so the labels can be checked by eye.

  python3 tools/build_video_ground_truth.py database/eval/video_3.npz \
      --video ../../EutPerceptionUtils/eut_utils/samples/video_3.mp4 --out-dir database/eval
"""

import argparse
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def build_tracklets(frame_idx, centers, dedupe_dist=0.02, link_dist=0.06, max_gap=15):
    reps = {}
    for f in np.unique(frame_idx):
        groups = []
        for i in np.where(frame_idx == f)[0]:
            for group in groups:
                if np.linalg.norm(centers[group[0]] - centers[i]) < dedupe_dist:
                    group.append(i)
                    break
            else:
                groups.append([i])
        reps[f] = [np.mean(centers[g], axis=0) for g in groups]

    tracks, active = [], []
    for f in sorted(reps):
        points = reps[f]
        used = set()
        if active:
            cost = np.array([[np.linalg.norm(tracks[t][max(tracks[t])] - p) + 0.01 * (f - max(tracks[t]))
                              for p in points] for t in active])
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < link_dist:
                    tracks[active[a]][f] = points[b]
                    used.add(b)
        for b, p in enumerate(points):
            if b not in used:
                tracks.append({f: p})
                active.append(len(tracks) - 1)
        active = [t for t in active if f - max(tracks[t]) <= max_gap]
    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--video', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--min-coverage', type=float, default=0.3, help='fraction of video frames a person track must cover')
    args = parser.parse_args()

    data = np.load(args.dataset)
    frame_idx, bbox = data['frame_idx'], data['bbox']
    centers = np.c_[(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2]
    tracks = build_tracklets(frame_idx, centers)
    n_frames = len(np.unique(frame_idx))
    people = [t for t, tr in enumerate(tracks) if len(tr) >= args.min_coverage * n_frames]

    labels = np.full(len(frame_idx), -1, dtype=np.int32)
    for i in range(len(frame_idx)):
        best = None
        for label, t in enumerate(people):
            point = tracks[t].get(frame_idx[i])
            if point is None:
                continue
            dist = np.linalg.norm(point - centers[i])
            if dist < 0.03 and (best is None or dist < best[0]):
                best = (dist, label)
        if best:
            labels[i] = best[1]

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.dataset))[0]
    np.save(os.path.join(args.out_dir, f'{stem}_gt.npy'), labels)
    print(f'{len(people)} people; label counts:', dict(zip(*np.unique(labels, return_counts=True))))

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    rows = []
    for label in list(range(len(people))) + [-1]:
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        tiles = []
        for i in idx[np.linspace(0, len(idx) - 1, 12).astype(int)]:
            img = frames[frame_idx[i]]
            h, w = img.shape[:2]
            x1, y1, x2, y2 = (bbox[i] * [w, h, w, h]).astype(int)
            tile = cv2.resize(img[max(0, y1):y2, max(0, x1):x2], (80, 80))
            cv2.putText(tile, str(frame_idx[i]), (2, 10), 0, 0.35, (0, 255, 255), 1)
            tiles.append(tile)
        row = np.hstack(tiles + [np.zeros((80, 80 * (12 - len(tiles)), 3), np.uint8)] if len(tiles) < 12 else tiles)
        cv2.putText(row, f'P{label}', (2, 78), 0, 0.5, (0, 0, 255), 2)
        rows.append(row)
    sheet = os.path.join(args.out_dir, f'{stem}_gt_sheet.jpg')
    cv2.imwrite(sheet, np.vstack(rows))
    print('contact sheet:', sheet)


if __name__ == '__main__':
    main()
