#!/usr/bin/env python3
"""
Replay stored identity galleries from a live session as scripted scenes.

Each scene lists (seconds, {track_id: identity_key}) segments; frames are drawn by
cycling that identity's stored embeddings. Expected groups say which stored
identities are the same person. Reports how many identities the manager ends with
per person and whether different people were merged.

  python3 tools/evaluate_live_fragments.py database/eval/live_2026-09-17_identities.npz
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_recognition.identity_manager import FaceIdentityManager  # noqa: E402

# 2026-09-17 RealSense session: U7/U13 = user, U8/U9/U10 = photos of his wife, U6 = user chin only.
PEOPLE = {'user': ['U7', 'U13'], 'wife': ['U8', 'U9', 'U10']}
SCENES = {
    # user seen, leaves, comes back looking different (U7 -> U13 in the live test)
    'user_changes_look': [(6.0, {'t1': 'U7'}), (3.0, {}), (6.0, {'t2': 'U13'}), (3.0, {}), (4.0, {'t3': 'U7'})],
    # three different photos of the wife shown one after another
    'wife_photos': [(6.0, {'t1': 'U8'}), (2.0, {}), (6.0, {'t2': 'U9'}), (2.0, {}), (6.0, {'t3': 'U10'})],
    # user and wife photo in view at the same time, then separately
    'user_and_wife_together': [(6.0, {'t1': 'U7', 't2': 'U8'}), (3.0, {}), (6.0, {'t3': 'U13'}), (6.0, {'t4': 'U9'})],
}


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **k: None


def run_scene(galleries, scene, params, fps=10.0):
    clock = {'t': 0.0}
    manager = FaceIdentityManager(logger=Quiet(), clock=lambda: clock['t'], **params)
    counters = {k: 0 for k in galleries}
    labels = {}  # stored key -> set of identities assigned
    for seconds, tracks in scene:
        for _ in range(int(seconds * fps)):
            clock['t'] += 1.0 / fps
            if not tracks:
                continue
            batch = {}
            for track_id, key in tracks.items():
                gallery = galleries[key]
                batch[track_id] = gallery[counters[key] % len(gallery)]
                counters[key] += 1
            result = manager.process_new_embedding_batch(batch, 1.0)
            for track_id, key in tracks.items():
                if result[track_id].unique_id:
                    labels.setdefault(key, []).append(result[track_id].unique_id)
    # Resolve labels through merges: a merged identity no longer exists
    final = set(manager.identity_clusters)
    last = {key: next((u for u in reversed(ids) if u in final), None) for key, ids in labels.items()}
    return manager, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identities')
    parser.add_argument('--params', default='{}')
    args = parser.parse_args()
    data = dict(np.load(args.identities))
    galleries = {k[:-len('_embeddings')]: data[k] / np.linalg.norm(data[k], axis=1, keepdims=True)
                 for k in data if k.endswith('_embeddings')}
    params = json.loads(args.params)
    report = {}
    for name, scene in SCENES.items():
        manager, last = run_scene(galleries, scene, params)
        people = {person: sorted({last.get(k) for k in keys if k in last}) for person, keys in PEOPLE.items()}
        present = {p: ids for p, ids in people.items() if ids}
        merged_people = [(a, b) for a, b in itertools.combinations(present, 2) if set(present[a]) & set(present[b])]
        report[name] = {'final_identities': len(manager.identity_clusters), 'created': manager.total_identities_created,
                        'seed_rematches': manager.total_seed_rematches, 'merges': manager.total_identity_merges,
                        'identities_per_person': {p: len(ids) for p, ids in present.items()},
                        'different_people_merged': merged_people, 'last_label': last}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
