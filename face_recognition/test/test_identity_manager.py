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
    manager.seed_match_threshold = 2.0  # force a duplicate instead of joining U1
    uid = None
    for _ in range(4):
        uid = (manager._seed('face_x', sample(rng, people[0]), clock.t, set()) or (uid, 0))[0]
    for _ in range(10):
        manager._add_embedding(manager.identity_clusters[uid], sample(rng, people[0]), clock.t)
    assert uid == 'U3'
    clock.t += 2.0  # next merge check
    feed(manager, clock, [{'face_1': sample(rng, people[1])}])
    assert 'U3' not in manager.identity_clusters
    assert 'U1' in manager.identity_clusters
    assert manager.total_identity_merges == 1


def test_confirmed_identities_persist_throttled_and_reload(rng, people):
    clock = Clock()
    store = FakeStore()
    manager = make(clock, store=store, persist_every=5, min_persist_interval=10.0, min_confirm_seconds=1.0,
                   redundancy_threshold=1.01)
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


def two_looks(rng, base, similarity=0.6):
    """Two appearance directions of one person whose cosine is about ``similarity``."""
    other = rng.normal(size=DIM)
    other -= (other @ base) * base
    other /= np.linalg.norm(other)
    angle = np.arccos(similarity) / 2
    look_a = np.cos(angle) * base + np.sin(angle) * other
    look_b = np.cos(angle) * base - np.sin(angle) * other
    return look_a, look_b


def test_person_returning_with_a_new_look_keeps_identity(rng, people):
    clock = Clock()
    manager = make(clock)
    look_a, look_b = two_looks(rng, people[0], similarity=0.62)
    feed(manager, clock, [{'face_0': sample(rng, look_a, 0.1)} for _ in range(40)])
    clock.t += 5.0
    result = feed(manager, clock, [{'face_9': sample(rng, look_b, 0.1)} for _ in range(40)])
    clock.t += 2.0
    result = feed(manager, clock, [{'face_9': sample(rng, look_b, 0.1)}])
    assert result['face_9'].unique_id == 'U1'
    assert len(manager.identity_clusters) == 1


def test_similar_people_seen_together_are_never_merged(rng, people):
    clock = Clock()
    manager = make(clock)
    look_a, look_b = two_looks(rng, people[0], similarity=0.62)  # lookalikes
    feed(manager, clock, [{'face_0': sample(rng, look_a, 0.1), 'face_1': sample(rng, look_b, 0.1)} for _ in range(40)])
    assert len(manager.identity_clusters) == 2
    ids = sorted(manager.identity_clusters)
    assert ids[1] in manager.identity_clusters[ids[0]].co_occurring
    clock.t += 5.0
    feed(manager, clock, [{'face_5': sample(rng, look_a, 0.1)} for _ in range(20)])
    clock.t += 2.0
    feed(manager, clock, [{'face_5': sample(rng, look_a, 0.1)}])
    assert len(manager.identity_clusters) == 2


def test_static_face_does_not_grow_gallery_or_writes(rng, people):
    clock = Clock()
    store = FakeStore()
    manager = make(clock, store=store, persist_every=5, min_persist_interval=1.0)
    frozen = sample(rng, people[0])
    feed(manager, clock, [{'face_0': frozen} for _ in range(200)])
    identity = manager.identity_clusters['U1']
    assert identity.confirmed
    assert len(identity.all_embeddings) == 1
    assert store.writes <= 2


def test_gallery_eviction_keeps_distinct_looks(rng, people):
    clock = Clock()
    manager = make(clock, max_embeddings_per_identity=10, redundancy_threshold=0.99)
    look_a, look_b = two_looks(rng, people[0], similarity=0.7)
    feed(manager, clock, [{'face_0': sample(rng, look_b, 0.05)} for _ in range(4)])
    feed(manager, clock, [{'face_0': sample(rng, look_a, 0.3)} for _ in range(60)])
    gallery = np.stack(manager.identity_clusters['U1'].all_embeddings)
    assert len(gallery) == 10
    assert np.max(gallery @ (look_b / np.linalg.norm(look_b))) > 0.95  # look B survived eviction


def test_five_point_alignment():
    from types import SimpleNamespace
    from face_recognition.face_alignment import TEMPLATE_112, align_face, five_points_from_msg
    width, height = 640, 480
    template = TEMPLATE_112 + np.array([200.0, 150.0], np.float32)  # face at a known place
    landmarks = [SimpleNamespace(x=0.0, y=0.0, c=0.0) for _ in range(70)]
    for index, point in zip((42, 39, 30, 54, 48), template):
        landmarks[index] = SimpleNamespace(x=point[0] / width, y=point[1] / height, c=1.0)
    msg = SimpleNamespace(landmarks=landmarks, width=width, height=height)
    points = five_points_from_msg(msg)
    assert np.allclose(points, template, atol=1e-3)
    image = np.zeros((height, width, 3), np.uint8)
    image[150:262, 200:312] = 255
    crop = align_face(image, points, size=112)
    assert crop.shape == (112, 112, 3)
    assert crop.mean() > 250  # the face region maps onto the whole crop
    landmarks[30] = SimpleNamespace(x=0.5, y=0.5, c=0.0)
    assert five_points_from_msg(msg) is None


def test_five_points_use_eye_contour_centre_when_available():
    from types import SimpleNamespace
    from face_recognition.face_alignment import five_points_from_msg
    width, height = 100, 100
    landmarks = [SimpleNamespace(x=0.0, y=0.0, c=0.0) for _ in range(70)]
    for index in (30, 54, 48):
        landmarks[index] = SimpleNamespace(x=0.5, y=0.8, c=1.0)
    # MediaPipe-style: inner corners at 39/42, full contours around centres (30, 40) and (70, 40)
    for index, cx in ((36, 30), (42, 70)):
        for k, (dx, dy) in enumerate([(-8, 0), (-4, -3), (4, -3), (8, 0), (4, 3), (-4, 3)]):
            landmarks[index + k] = SimpleNamespace(x=(cx + dx) / width, y=(40 + dy) / height, c=1.0)
    msg = SimpleNamespace(landmarks=landmarks, width=width, height=height)
    points = five_points_from_msg(msg)
    assert np.allclose(points[0], (30, 40), atol=1e-3)
    assert np.allclose(points[1], (70, 40), atol=1e-3)


def test_new_identities_never_reuse_stored_numbers(rng, people):
    clock = Clock()
    store = FakeStore()
    store.max_user_number = lambda: 13  # e.g. U13 stored under another model key
    manager = make(clock, store=store)
    feed(manager, clock, [{'face_0': sample(rng, people[0])} for _ in range(4)])
    assert list(manager.identity_clusters) == ['U14']
