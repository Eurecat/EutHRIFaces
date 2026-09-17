"""FaceIdentityManager rules, without ROS: `python3 -m pytest test/test_identity_manager.py`."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_recognition.face_quality import FaceQualityConfig, face_quality  # noqa: E402
from face_recognition.identity_manager import (  # noqa: E402
    STATUS_CONFIRMED, STATUS_TENTATIVE, STATUS_UNKNOWN, FaceIdentityManager)

DIM = 64


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **k: None


class FakeStore:
    def __init__(self, fail=False):
        self.docs = {}
        self.writes = 0
        self.fail = fail

    def load(self):
        import copy
        return [copy.deepcopy(d) for d in self.docs.values()]

    def save(self, identity):
        import copy
        if self.fail:
            raise ConnectionError('mongo down')
        self.docs[identity.unique_id] = copy.deepcopy(identity)
        self.writes += 1

    def delete(self, unique_id):
        self.docs.pop(unique_id, None)

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def people(rng):
    """Five well separated identity directions."""
    q, _ = np.linalg.qr(rng.normal(size=(DIM, DIM)))
    return [q[:, i] for i in range(5)]


def sample(rng, base, noise=0.25):
    v = base + noise * rng.normal(size=DIM) / np.sqrt(DIM)
    return v / np.linalg.norm(v)


def make(clock, store=None, **kw):
    return FaceIdentityManager(logger=Quiet(), store=store, clock=clock, **kw)


def feed(manager, clock, frames, dt=0.15):
    """frames: list of {track_id: embedding} or ({track: emb}, {track: quality})."""
    out = None
    for frame in frames:
        embs, quality = frame if isinstance(frame, tuple) else (frame, 1.0)
        clock.t += dt
        out = manager.process_new_embedding_batch(embs, quality)
    return out


def test_single_sample_does_not_create_identity(rng, people):
    clock = Clock()
    manager = make(clock)
    result = feed(manager, clock, [{'face_0': sample(rng, people[0])}])
    assert result['face_0'].unique_id is None
    assert result['face_0'].status == STATUS_UNKNOWN
    assert manager.identity_clusters == {}


def test_consistent_seed_creates_then_confirms(rng, people):
    clock = Clock()
    manager = make(clock, min_seed_samples=4, min_confirm_samples=12, min_confirm_seconds=1.0)
    result = feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(4)])
    assert result['face_0'] == (result['face_0'].unique_id, result['face_0'].confidence, STATUS_TENTATIVE)
    assert result['face_0'].unique_id == 'U1'
    result = feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(10)])
    assert result['face_0'].unique_id == 'U1'
    assert result['face_0'].status == STATUS_CONFIRMED


def test_poor_quality_faces_never_create_or_teach(rng, people):
    clock = Clock()
    manager = make(clock)
    feed(manager, clock, [({'face_0': sample(rng, people[0])}, {'face_0': 0.0}) for _ in range(20)])
    assert manager.identity_clusters == {}

    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(20)])
    samples_before = manager.identity_clusters['U1'].good_samples
    result = feed(manager, clock, [({'face_0': sample(rng, people[0])}, {'face_0': 0.1}) for _ in range(5)])
    assert result['face_0'].unique_id == 'U1'  # still matched to the known identity
    assert manager.identity_clusters['U1'].good_samples == samples_before


def test_track_mixing_two_people_does_not_seed(rng, people):
    clock = Clock()
    manager = make(clock)
    frames = [{'face_0': sample(rng, people[i % 2])} for i in range(12)]
    feed(manager, clock, frames)
    assert manager.identity_clusters == {}


def test_known_identity_recognized_on_new_track(rng, people):
    clock = Clock()
    manager = make(clock)
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(20)])
    clock.t += 10.0
    result = feed(manager, clock, [{'face_9': sample(rng, people[0])}])
    assert result['face_9'].unique_id == 'U1'
    assert manager.total_identities_created == 1


def test_exclusive_assignment_within_frame(rng, people):
    clock = Clock()
    manager = make(clock)
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(20)])
    result = feed(manager, clock, [{'face_1': sample(rng, people[0]), 'face_2': sample(rng, people[0])}])
    ids = [result['face_1'].unique_id, result['face_2'].unique_id]
    assert ids.count('U1') == 1


def test_ambiguous_match_stays_unlabeled(rng, people):
    clock = Clock()
    manager = make(clock, min_seed_samples=4)
    feed(manager, clock, [{'face_0': sample(rng, people[0]), 'face_1': sample(rng, people[1])} for _ in range(20)])
    between = (people[0] + people[1]) / np.linalg.norm(people[0] + people[1])
    result = feed(manager, clock, [({'face_5': between}, {'face_5': 0.0})])
    assert result['face_5'].unique_id is None


def test_many_people_many_loops_stay_five_identities(rng, people):
    clock = Clock()
    manager = make(clock)
    for loop in range(5):
        # tracker ids rotate every loop, like ByteTrack at a video wrap
        frames = [{f'face_{(p + loop) % 5}': sample(rng, people[p]) for p in range(5)} for _ in range(30)]
        feed(manager, clock, frames)
        clock.t += 3.0
    assert len(manager.identity_clusters) == 5
    assert manager.total_identities_created == 5


def test_fragments_are_merged_keeping_lowest_number(rng, people):
    clock = Clock()
    manager = make(clock, merge_threshold=0.7)
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(20)])
    feed(manager, clock, [{'face_1': sample(rng, people[1])} for _ in range(20)])
    # Force a duplicate of person 0 as U3
    manager._seed_buffers.clear()
    uid = manager._seed('face_x', sample(rng, people[0]), clock.t)
    for _ in range(3):
        uid = manager._seed('face_x', sample(rng, people[0]), clock.t) or uid
    for _ in range(10):
        manager._add_embedding(manager.identity_clusters[uid], sample(rng, people[0]), clock.t)
    assert uid == 'U3'
    feed(manager, clock, [{'face_1': sample(rng, people[1])}])
    assert 'U3' not in manager.identity_clusters
    assert 'U1' in manager.identity_clusters
    assert manager.total_identity_merges == 1


def test_confirmed_identities_persist_throttled_and_reload(rng, people):
    clock = Clock()
    store = FakeStore()
    manager = make(clock, store=store, persist_every=5, min_persist_interval=10.0)
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(11)])
    assert 'U1' not in store.docs  # tentative identities are never written
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(11)])
    assert 'U1' in store.docs
    writes = store.writes
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(20)], dt=0.1)  # 2 s, 20 updates
    assert store.writes == writes  # throttled by min_persist_interval
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(6)], dt=2.0)
    assert store.writes > writes

    feed(manager, clock, [{'face_1': sample(rng, people[1])} for _ in range(25)])
    manager.flush(force=True)
    restarted = make(clock, store=store)
    assert set(restarted.identity_clusters) == {'U1', 'U2'}
    clock.t += 60.0
    result = feed(restarted, clock, [{'face_7': sample(rng, people[1])}])
    assert result['face_7'] == ('U2', result['face_7'].confidence, STATUS_CONFIRMED)
    feed(restarted, clock, [{'face_8': sample(rng, people[2])} for _ in range(4)])
    assert 'U3' in restarted.identity_clusters


def test_store_failure_does_not_stop_recognition(rng, people):
    clock = Clock()
    manager = make(clock, store=FakeStore(fail=True))
    result = feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(25)])
    assert result['face_0'].unique_id == 'U1'
    assert not manager.identity_clusters['U1'].persisted


def test_tentative_identities_expire_confirmed_ones_stay(rng, people):
    clock = Clock()
    manager = make(clock, identity_timeout=30.0)
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(25)])
    feed(manager, clock, [{'face_1': sample(rng, people[1])} for _ in range(4)])
    assert manager.identity_clusters['U1'].confirmed
    assert not manager.identity_clusters['U2'].confirmed
    clock.t += 31.0
    feed(manager, clock, [{'face_9': sample(rng, people[3])}])
    assert 'U1' in manager.identity_clusters
    assert 'U2' not in manager.identity_clusters


def test_stale_track_mapping_forgotten(rng, people):
    clock = Clock()
    manager = make(clock, track_timeout=2.0)
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(20)])
    clock.t += 5.0
    feed(manager, clock, [{'face_3': sample(rng, people[1])}])
    assert 'face_0' not in manager.track_id_to_unique_id


def test_face_quality():
    config = FaceQualityConfig(min_detection_confidence=0.4, profile_eye_ratio=0.10, frontal_eye_ratio=0.18)
    box = (0.40, 0.20, 0.50, 0.35)  # width 0.10
    assert face_quality(box, 0.9, 0.42, 0.44, config) == pytest.approx(1.0)   # ratio 0.20
    assert face_quality(box, 0.9, 0.43, 0.444, config) == pytest.approx(0.5)  # ratio 0.14
    assert face_quality(box, 0.9, 0.43, 0.435, config) == 0.0                 # profile
    assert face_quality(box, 0.2, 0.42, 0.44, config) == 0.0                  # low detection confidence
    assert face_quality(box, 0.9, None, 0.44, config) == 0.0
