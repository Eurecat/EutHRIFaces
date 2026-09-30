#!/usr/bin/env python3
"""Enroll a synthetic face identity into the shared database, as one robot would.

This exists so the cross-robot gallery refresh (scenario D) can be driven without a camera:
run it in robot A's container to make A enroll a profile, then watch robot B adopt it within
its refresh interval. The embedding is random, so it is deliberately *not* matchable by any
face in a video - it tests propagation and adoption, not recognition.

Usage, inside a face_recognition container:

    /opt/ros_python_env/bin/python /workspace/src/face_recognition/tools/\\
        enroll_test_identity.py --unique-id U2

The database and the profile scope come from the same environment variables the node uses
(``FACE_DB_MONGO_URI`` / ``FACE_PROFILE_SCOPE``), so what this writes lands in exactly the
gallery the node reads.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.identity_manager import (  # noqa: E402
    FaceIdentityCluster,
    MongoFaceIdentityStore,
    normalize_embedding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-id", required=True,
                        help="Identity to create, e.g. U2. Must be free in the gallery.")
    parser.add_argument("--dimensions", type=int, default=512,
                        help="Embedding size (default 512, matching vggface2/arcface)")
    parser.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", ""),
                        help="Robot recorded as the author (default $ROBOT_ID)")
    parser.add_argument("--mongo-uri", default=os.environ.get("FACE_DB_MONGO_URI", ""),
                        help="Database to write to (default $FACE_DB_MONGO_URI)")
    parser.add_argument("--scope", default=os.environ.get("FACE_PROFILE_SCOPE", ""),
                        help="Profile scope / model_key (default $FACE_PROFILE_SCOPE)")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic embedding seed")
    args = parser.parse_args()

    if not args.mongo_uri:
        print("No mongo URI: set --mongo-uri or FACE_DB_MONGO_URI", file=sys.stderr)
        return 2
    if not args.scope:
        print("No profile scope: set --scope or FACE_PROFILE_SCOPE", file=sys.stderr)
        return 2

    store = MongoFaceIdentityStore(args.mongo_uri, model_key=args.scope, robot_id=args.robot_id)
    existing = {identity.unique_id for identity in store.load()}
    if args.unique_id in existing:
        print(f"{args.unique_id} already exists in scope {args.scope}; nothing to do")
        store.close()
        return 0

    random = np.random.RandomState(args.seed)
    vector = normalize_embedding(random.randn(args.dimensions).astype(np.float32))
    now = time.time()
    identity = FaceIdentityCluster(
        unique_id=args.unique_id,
        creation_timestamp=now,
        last_seen_timestamp=now,
        all_embeddings=[vector],
        mean_embedding=vector,
        confirmed=True,
        good_samples=20,
        total_detections=20,
    )
    store.save(identity)
    store.close()

    # The timestamp is the point of the tool: it lets a test measure how long another robot
    # takes to adopt this identity through its own refresh poll.
    print(f"ENROLLED unique_id={args.unique_id} scope={args.scope} robot={args.robot_id or '-'} "
          f"at {now:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
