"""Cross-robot gallery refresh: adoption, union-merge, tombstones, watermarks.

These tests use an in-memory stand-in for :class:`MongoFaceIdentityStore` so they run without
a database. The store contract the refresh relies on is small and explicit:
``changed_since(watermark)``, ``load()``, ``save()``, ``max_user_number()``, ``mark_merged()``.
"""

import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from face_recognition.identity_manager import (  # noqa: E402
    FaceIdentityCluster,
    FaceIdentityManager,
    MongoFaceIdentityStore,
    normalize_embedding,
)


class Clock:
    """Advanceable clock, so watermarks and timestamps are deterministic."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def vector(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return normalize_embedding(np.asarray([x, y, z], dtype=np.float32))


class FakeStore:
    """In-memory face identity store, driven explicitly by the test."""

    def __init__(self, clock: Clock, model_key: str = "vggface2-aligned"):
        self.clock = clock
        self.model_key = model_key
        self.documents = {}        # unique_id -> document (what the shared DB holds)
        self.saved = []            # unique_ids this process wrote
        self.merged = []           # (unique_id, merged_into) tombstone writes
        self.watermarks = []       # watermark of every changed_since call
        self.fail_now = False

    # -- test helpers ------------------------------------------------------
    def publish(self, unique_id: str, embeddings, *, revision: int = 1,
                confirmed: bool = True, merged_into: str = None, updated_by: str = "robot_a"):
        """Write a document the way another robot's store would."""
        vectors = [vector(*e) if not isinstance(e, np.ndarray) else e for e in embeddings]
        self.documents[unique_id] = {
            "model_key": self.model_key,
            "unique_id": unique_id,
            "mean_embedding": vectors[0].astype(float).tolist() if vectors else None,
            "embeddings": [v.astype(float).tolist() for v in vectors],
            "confirmed": confirmed,
            "revision": revision,
            "updated_at": self.clock.now,
            "updated_by_robot": updated_by,
            "total_detections": 5,
            "good_samples": 5,
            "creation_timestamp": self.clock.now - 100.0,
            "last_seen_timestamp": self.clock.now,
            **({"merged_into": merged_into} if merged_into else {}),
        }

    # -- store protocol used by FaceIdentityManager -------------------------
    def max_user_number(self) -> int:
        numbers = [int(m.group(1)) for uid in self.documents
                   for m in [re.search(r"(\d+)$", uid)] if m]
        return max(numbers or [0])

    def load(self):
        clusters = []
        for document in self.documents.values():
            if "merged_into" in document:
                continue
            embeddings = [normalize_embedding(np.asarray(e, dtype=np.float32))
                          for e in document["embeddings"]]
            cluster = FaceIdentityCluster(
                unique_id=document["unique_id"],
                creation_timestamp=document["creation_timestamp"],
                last_seen_timestamp=document["last_seen_timestamp"],
                all_embeddings=embeddings,
                mean_embedding=normalize_embedding(np.asarray(document["mean_embedding"], dtype=np.float32)),
                confirmed=document["confirmed"],
                revision=document["revision"],
            )
            clusters.append(cluster)
        return clusters

    def changed_since(self, watermark: float):
        self.watermarks.append(watermark)
        if self.fail_now:
            raise RuntimeError("database unavailable")
        return [document for _, document in sorted(self.documents.items())
                if float(document.get("updated_at", 0.0)) > float(watermark)]

    def save(self, identity: FaceIdentityCluster) -> None:
        self.saved.append(identity.unique_id)
        identity.revision += 1
        self.publish(identity.unique_id,
                     [e for e in identity.all_embeddings],
                     revision=identity.revision,
                     updated_by="self")

    def mark_merged(self, unique_id: str, merged_into: str) -> bool:
        self.merged.append((unique_id, merged_into))
        if unique_id in self.documents:
            self.documents[unique_id]["merged_into"] = merged_into
            self.documents[unique_id]["revision"] += 1
            return True
        return False

    def delete(self, unique_id: str) -> None:
        self.documents.pop(unique_id, None)

    def close(self) -> None:
        pass


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock):
    return FakeStore(clock)


@pytest.fixture
def manager(clock, store):
    return FaceIdentityManager(logger=None, store=store, clock=clock)


# ---------------------------------------------------------------------------
# Adoption (scenario D)
# ---------------------------------------------------------------------------

def test_robot_b_adopts_an_identity_robot_a_enrolled(clock, store, manager):
    """The whole point of the refresh: an identity this robot never saw becomes usable."""
    clock.advance(5.0)
    store.publish("U7", [(1.0, 0.0, 0.0)], revision=3)

    counters = manager.refresh_from_store()

    assert counters["adopted"] == 1
    assert "U7" in manager.identity_clusters
    adopted = manager.identity_clusters["U7"]
    assert adopted.revision == 3
    assert adopted.confirmed is True
    assert adopted.persisted is True, "it already lives in the shared database"
    assert adopted.all_embeddings, "the gallery comes with it, so it can be matched"
    # And it is usable: the embedding it was enrolled with scores as a match.
    assert manager.score("U7", vector(1.0, 0.0, 0.0)) is not None


def test_adoption_keeps_the_next_free_user_number(clock, store, manager):
    clock.advance(5.0)
    store.publish("U9", [(1.0, 0.0, 0.0)])
    manager.refresh_from_store()
    assert manager._next_user_number == 10


def test_refresh_is_idempotent_and_does_not_write_back(clock, store, manager):
    """Polling the same unchanged document twice must not cause writes (no revision storm)."""
    clock.advance(5.0)
    store.publish("U7", [(1.0, 0.0, 0.0)], revision=3)

    first = manager.refresh_from_store()
    store.saved.clear()
    second = manager.refresh_from_store()

    assert first["adopted"] == 1
    assert second["adopted"] == 0 and second["updated"] == 0
    assert store.saved == []
    assert store.merged == []


def test_a_newer_peer_revision_unions_the_galleries(clock, store, manager):
    """Two robots holding one identity must keep both robots' embeddings."""
    manager.identity_clusters["U1"] = FaceIdentityCluster(
        unique_id="U1", creation_timestamp=clock.now - 50, last_seen_timestamp=clock.now,
        all_embeddings=[vector(1.0, 0.0, 0.0)], mean_embedding=vector(1.0, 0.0, 0.0),
        confirmed=True, persisted=True, revision=1)
    clock.advance(5.0)
    store.publish("U1", [(0.0, 1.0, 0.0)], revision=4)

    counters = manager.refresh_from_store()

    identity = manager.identity_clusters["U1"]
    assert counters["updated"] == 1
    assert counters["embeddings_added"] == 1
    assert identity.revision == 4
    gallery = np.stack(identity.all_embeddings)
    assert len(gallery) == 2, "the peer's look is added, ours is kept"
    assert any(float(v @ vector(0.0, 1.0, 0.0)) > 0.99 for v in gallery)


def test_union_with_nothing_new_does_not_mark_the_identity_dirty(clock, store, manager):
    """Convergence, not ping-pong: merging an identical gallery must not trigger a write."""
    manager.identity_clusters["U1"] = FaceIdentityCluster(
        unique_id="U1", creation_timestamp=clock.now - 50, last_seen_timestamp=clock.now,
        all_embeddings=[vector(1.0, 0.0, 0.0)], mean_embedding=vector(1.0, 0.0, 0.0),
        confirmed=True, persisted=True, revision=1)
    clock.advance(5.0)
    store.publish("U1", [(1.0, 0.0, 0.0)], revision=4)

    manager.refresh_from_store()

    assert manager.identity_clusters["U1"].unsaved_updates == 0
    manager.flush()
    assert store.saved == []


def test_an_older_peer_revision_does_not_overwrite_ours(clock, store, manager):
    manager.identity_clusters["U1"] = FaceIdentityCluster(
        unique_id="U1", creation_timestamp=clock.now - 50, last_seen_timestamp=clock.now,
        all_embeddings=[vector(1.0, 0.0, 0.0), vector(0.0, 1.0, 0.0)],
        mean_embedding=vector(1.0, 0.0, 0.0),
        confirmed=True, persisted=True, revision=7)
    clock.advance(5.0)
    store.publish("U1", [(0.0, 1.0, 0.0)], revision=2)

    counters = manager.refresh_from_store()

    identity = manager.identity_clusters["U1"]
    assert counters["updated"] == 0
    assert identity.revision == 7
    assert len(identity.all_embeddings) == 2


def test_a_locally_confirmed_identity_survives_a_peer_that_lacks_it(clock, store, manager):
    """A peer that has only just started must not be able to delete our identities."""
    manager.identity_clusters["U1"] = FaceIdentityCluster(
        unique_id="U1", creation_timestamp=clock.now - 50, last_seen_timestamp=clock.now,
        all_embeddings=[vector(1.0, 0.0, 0.0)], mean_embedding=vector(1.0, 0.0, 0.0),
        confirmed=True, persisted=True, revision=1)
    clock.advance(5.0)
    store.publish("U2", [(0.0, 1.0, 0.0)])

    manager.refresh_from_store()

    assert "U1" in manager.identity_clusters


# ---------------------------------------------------------------------------
# Tombstones (a peer merged two identities)
# ---------------------------------------------------------------------------

def test_a_peer_tombstone_retires_the_merged_identity(clock, store, manager):
    manager.identity_clusters["U1"] = FaceIdentityCluster(
        unique_id="U1", creation_timestamp=clock.now - 50, last_seen_timestamp=clock.now,
        all_embeddings=[vector(1.0, 0.0, 0.0)], mean_embedding=vector(1.0, 0.0, 0.0),
        confirmed=True, persisted=True, revision=1)
    manager.identity_clusters["U2"] = FaceIdentityCluster(
        unique_id="U2", creation_timestamp=clock.now - 40, last_seen_timestamp=clock.now,
        all_embeddings=[vector(0.0, 1.0, 0.0)], mean_embedding=vector(0.0, 1.0, 0.0),
        confirmed=True, persisted=True, revision=1)
    manager.track_id_to_unique_id["face_1"] = "U2"
    clock.advance(5.0)
    store.publish("U2", [], revision=2, merged_into="U1")

    counters = manager.refresh_from_store()

    assert counters["retired"] == 1
    assert "U2" not in manager.identity_clusters
    assert manager.track_id_to_unique_id["face_1"] == "U1", "the live track follows the survivor"
    gallery = manager.identity_clusters["U1"].all_embeddings
    assert len(gallery) == 2, "the retired identity's gallery is folded into the survivor"


def test_a_tombstone_for_an_identity_we_never_had_is_a_no_op(clock, store, manager):
    clock.advance(5.0)
    store.publish("U5", [], revision=2, merged_into="U1")

    counters = manager.refresh_from_store()

    assert counters["retired"] == 0
    assert manager.identity_clusters == {}


# ---------------------------------------------------------------------------
# Watermark and failure behaviour
# ---------------------------------------------------------------------------

def test_the_watermark_only_moves_forward(clock, store, manager):
    clock.advance(10.0)
    manager.refresh_from_store()
    first = store.watermarks[-1]
    clock.advance(30.0)
    manager.refresh_from_store()
    second = store.watermarks[-1]

    assert first > 0.0, "the first poll does not replay the whole database"
    assert second > first


def test_the_poll_window_is_bounded_and_documents_are_not_re_read_forever(clock, store, manager):
    """Only the recent slack window is ever re-read; older documents drop out of the poll.

    The watermark is deliberately set to ``query_start - slack`` so that a document written
    while the poll was running (by a robot whose clock is slightly behind ours) is not skipped.
    The price is that documents written within that window are looked at once more, which is
    harmless - adoption is guarded by the revision - and stops after the next poll.
    """
    store.publish("U1", [(1.0, 0.0, 0.0)])
    manager.refresh_from_store()
    clock.advance(60.0)
    store.publish("U2", [(0.0, 1.0, 0.0)], updated_by="robot_b")

    second = manager.refresh_from_store()
    assert second["adopted"] == 1, "the new document is adopted"
    assert second["updated"] == 0

    # The re-read window is bounded: the document falls out for good after the next interval,
    # and none of this caused a write.
    clock.advance(60.0)
    assert manager.refresh_from_store()["adopted"] == 0
    clock.advance(60.0)
    assert manager.refresh_from_store()["polled"] == 0, "the poll is quiet again"
    assert store.saved == []


def test_a_database_failure_raises_so_the_node_can_back_off(clock, store, manager):
    store.fail_now = True

    with pytest.raises(RuntimeError):
        manager.refresh_from_store()

    assert manager.total_refresh_failures == 1
    store.fail_now = False
    manager._clock.advance(5.0)
    assert manager.refresh_from_store() == {"polled": 0, "adopted": 0, "updated": 0,
                                            "retired": 0, "embeddings_added": 0}


def test_refresh_without_a_store_is_a_no_op(clock):
    manager = FaceIdentityManager(logger=None, store=None, clock=clock)
    assert manager.refresh_from_store() == {}


# ---------------------------------------------------------------------------
# Store side: what gets written and what gets loaded
# ---------------------------------------------------------------------------

class RecordingCursor(list):
    """pymongo's find() returns a cursor, whose sort() takes (key, direction) positionally."""

    def sort(self, key_or_list, direction=None):
        ascending = direction is None or str(direction).lower() in ("1", "asc", "ascending")
        return RecordingCursor(sorted(self, key=lambda document: document.get(key_or_list, 0),
                                      reverse=not ascending))


class RecordingCollection:
    """Just enough of a pymongo collection to inspect the queries the store builds."""

    def __init__(self, documents=()):
        self.documents = [dict(document) for document in documents]
        self.find_filters = []
        self.update_calls = []

    def _matches(self, document, query):
        for key, expected in (query or {}).items():
            value = document.get(key)
            if isinstance(expected, dict):
                if "$exists" in expected and (key in document) != bool(expected["$exists"]):
                    return False
                if "$gt" in expected and not (value is not None and float(value) > float(expected["$gt"])):
                    return False
                if "$ne" in expected and value == expected["$ne"]:
                    return False
            elif value != expected:
                return False
        return True

    def find(self, query=None, projection=None):
        self.find_filters.append(query)
        return RecordingCursor(document for document in self.documents
                               if self._matches(document, query))

    def update_one(self, query, update, **kwargs):
        self.update_calls.append((query, update, kwargs))

        class Result:
            matched_count = 1
        return Result()

    def count_documents(self, query=None):
        return sum(1 for document in self.documents if self._matches(document, query))

    def create_index(self, *args, **kwargs):
        return "index"


def store_with(collection) -> MongoFaceIdentityStore:
    """Build the store without a server: the refresh only touches these three attributes."""
    store = object.__new__(MongoFaceIdentityStore)
    store._collection = collection
    store._model_key = "vggface2-aligned"
    store._robot_id = "robot_a"
    store._save_last_n = 100
    return store


def test_changed_since_is_scoped_to_the_gallery_and_ordered():
    collection = RecordingCollection([
        {"unique_id": "U1", "model_key": "vggface2-aligned", "updated_at": 10.0},
        {"unique_id": "U2", "model_key": "vggface2-aligned", "updated_at": 30.0},
        {"unique_id": "U3", "model_key": "other-model", "updated_at": 30.0},
    ])
    documents = store_with(collection).changed_since(20.0)

    assert [document["unique_id"] for document in documents] == ["U2"]
    assert collection.find_filters[-1] == {"model_key": "vggface2-aligned",
                                           "updated_at": {"$gt": 20.0}}


def test_load_ignores_tombstoned_identities():
    """A robot that restarts must not bring back an id the fleet already retired."""
    collection = RecordingCollection([
        {"unique_id": "U1", "model_key": "vggface2-aligned", "embeddings": [[1.0, 0.0]],
         "mean_embedding": [1.0, 0.0]},
        {"unique_id": "U2", "model_key": "vggface2-aligned", "embeddings": [],
         "mean_embedding": None, "merged_into": "U1"},
    ])
    loaded = store_with(collection).load()

    assert [identity.unique_id for identity in loaded] == ["U1"]
    assert collection.find_filters[-1] == {"model_key": "vggface2-aligned",
                                           "merged_into": {"$exists": False}}


def test_load_reads_back_the_revision():
    collection = RecordingCollection([
        {"unique_id": "U1", "model_key": "vggface2-aligned", "embeddings": [[1.0, 0.0]],
         "mean_embedding": [1.0, 0.0], "revision": 6},
    ])
    assert store_with(collection).load()[0].revision == 6


def test_mark_merged_tombstones_instead_of_deleting():
    collection = RecordingCollection([
        {"unique_id": "U2", "model_key": "vggface2-aligned"},
    ])

    assert store_with(collection).mark_merged("U2", "U1") is True

    query, update, _ = collection.update_calls[-1]
    assert query == {"model_key": "vggface2-aligned", "unique_id": "U2"}
    assert update["$set"]["merged_into"] == "U1"
    assert update["$set"]["updated_by_robot"] == "robot_a"
    assert update["$inc"] == {"revision": 1}, "the bump is what makes other robots notice it"
    assert update["$set"]["embeddings"] == [], "the survivor already absorbed the gallery"
    assert update["$set"]["mean_embedding"] is None
