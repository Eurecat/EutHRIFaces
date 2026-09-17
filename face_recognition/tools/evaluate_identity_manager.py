#!/usr/bin/env python3
"""
Replay a recorded face dataset through FaceIdentityManager and score it against labels.

Deterministic and fast (no ROS, no GPU): use it to tune identity parameters.

  python3 tools/evaluate_identity_manager.py database/eval/video_3.npz database/eval/video_3_gt.npy \
      --loops 3 --restart

Metrics:
  created / final / persisted   identities created, alive at the end, saved to the (fake) store
  purity                        labeled faces whose identity's dominant person is their person
  ids_per_person                distinct identities that labeled each person (1 is ideal)
  coverage                      fraction of each person's faces that got a label
  fp_labeled                    false-positive detections (label -1) that got a label
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_recognition.face_quality import face_quality, RIGHT_EYE_INSIDE, LEFT_EYE_INSIDE  # noqa: E402
from face_recognition.identity_manager import FaceIdentityManager  # noqa: E402


class MemoryStore:
    def __init__(self):
        self.docs = {}
        self.writes = 0

    def load(self):
        return [self.docs[k] for k in sorted(self.docs)]

    def save(self, identity):
        import copy
        self.docs[identity.unique_id] = copy.deepcopy(identity)
        self.writes += 1

    def delete(self, unique_id):
        self.docs.pop(unique_id, None)

    def close(self):
        pass


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **k: None


def qualities(data):
    lm = data['landmarks']
    return np.array([
        face_quality(data['bbox'][i], float(data['bbox_conf'][i]),
                     lm[i, LEFT_EYE_INSIDE, 0] if lm[i, LEFT_EYE_INSIDE, 2] > 0 else None,
                     lm[i, RIGHT_EYE_INSIDE, 0] if lm[i, RIGHT_EYE_INSIDE, 2] > 0 else None)
        for i in range(len(lm))])


def replay(data, labels, embedding_key='emb_matched', loops=1, restart=False, params=None):
    params = params or {}
    quality = qualities(data)
    order = np.argsort(data['stamp_ns'], kind='stable')
    stamps = data['stamp_ns'][order]
    t0 = stamps[0]
    span = (stamps[-1] - t0) / 1e9 + 1.0
    clock = {'t': 0.0}
    store = MemoryStore()

    def make():
        return FaceIdentityManager(logger=Quiet(), store=store, clock=lambda: clock['t'], **params)

    manager = make()
    created = 0
    assign = {}
    for loop in range(loops):
        if restart and loop > 0:
            created += manager.total_identities_created
            manager = make()
        for stamp in np.unique(stamps):
            rows = order[stamps == stamp]
            clock['t'] = loop * span + (stamp - t0) / 1e9
            result = manager.process_new_embedding_batch(
                {str(data['face_id'][r]): data[embedding_key][r] for r in rows},
                {str(data['face_id'][r]): float(quality[r]) for r in rows})
            for r in rows:
                a = result.get(str(data['face_id'][r]))
                assign[r] = a.unique_id if a else None
    manager.flush(force=True)
    created += manager.total_identities_created
    return score(assign, labels, created, manager, store)


def score(assign, labels, created, manager, store):
    person_rows = [i for i in range(len(labels)) if labels[i] >= 0]
    labeled = [i for i in person_rows if assign.get(i)]
    by_identity = collections.defaultdict(list)
    for i in labeled:
        by_identity[assign[i]].append(labels[i])
    correct = sum(collections.Counter(v).most_common(1)[0][1] for v in by_identity.values())
    people = sorted({int(labels[i]) for i in person_rows})
    return {
        'created': int(created),
        'final': len(manager.identity_clusters),
        'confirmed': sum(c.confirmed for c in manager.identity_clusters.values()),
        'persisted': len(store.docs),
        'store_writes': store.writes,
        'purity': round(correct / max(1, len(labeled)), 3),
        'labeled': round(len(labeled) / max(1, len(person_rows)), 3),
        'ids_per_person': {p: len({assign[i] for i in labeled if labels[i] == p}) for p in people},
        'coverage': {p: round(sum(1 for i in person_rows if labels[i] == p and assign.get(i))
                              / max(1, int(np.sum(labels == p))), 2) for p in people},
        'fp_labeled': f"{sum(1 for i in np.where(labels < 0)[0] if assign.get(i))}/{int(np.sum(labels < 0))}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('labels')
    parser.add_argument('--embedding', default='emb_matched', choices=['emb_matched', 'emb_latest'])
    parser.add_argument('--loops', type=int, default=3)
    parser.add_argument('--restart', action='store_true', help='recreate the manager from the store every loop')
    parser.add_argument('--params', default='{}', help='JSON overrides for FaceIdentityManager')
    args = parser.parse_args()

    data = dict(np.load(args.dataset))
    labels = np.load(args.labels)
    result = replay(data, labels, args.embedding, args.loops, args.restart, json.loads(args.params))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
