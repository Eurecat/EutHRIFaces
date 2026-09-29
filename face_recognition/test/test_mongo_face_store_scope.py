"""Profile scope, provenance and revision of the persisted face gallery.

The MongoDB store is driven through a minimal fake pymongo module, so these tests pin the
*shape* of the writes (which scope key a document belongs to, which robot is recorded as
its author, and that ``revision`` grows) without needing a server. That contract is what
lets several robots share one database safely, so it is worth asserting exactly.

Run: ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest test/test_mongo_face_store_scope.py``
"""

import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_recognition.identity_manager import (  # noqa: E402
    FaceIdentityCluster, MongoFaceIdentityStore, normalize_embedding)

DIM = 64


class FakeCollection:
    """Records the calls the store makes; emulates the parts of the Mongo API it uses."""

    def __init__(self):
        self.updates = []
        self.deletes = []
        self.indexes = []
        self.documents = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))
        return 'index'

    def count_documents(self, query):
        return len(self.documents)

    def find(self, query, projection=None):
        return iter(self.documents)

    def update_one(self, query, update, upsert=False):
        self.updates.append((dict(query), update, upsert))
        return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)

    def find_one_and_update(self, query, update, upsert=False, return_document=False):
        """Recorded exactly like update_one; returns the post-update document, as pymongo does.

        The store uses find_one_and_update so it can learn the revision the database actually
        holds (needed to tell its own copy from a peer's during a shared-gallery refresh).
        """
        self.updates.append((dict(query), update, upsert))
        revision = int(update.get('$inc', {}).get('revision', 0))
        return {**dict(query), **update.get('$set', {}), 'revision': revision}

    def delete_one(self, query):
        self.deletes.append(dict(query))
        return SimpleNamespace(deleted_count=1)


class FakeDatabase:
    """`client[db][collection]` — both levels return the same recording collection."""

    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        return self.collection


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.uri = args[0] if args else None
        self.collection = FakeCollection()
        self.admin = SimpleNamespace(command=lambda *a, **k: {'ok': 1})

    def __getitem__(self, database_name):
        return FakeDatabase(self.collection)

    def close(self):
        pass


@pytest.fixture
def store_factory(monkeypatch):
    """Build a store against a fake pymongo module; returns (store, collection)."""
    created = []

    def make(model_key, robot_id=''):
        collection = FakeCollection()
        client = FakeClient('mongodb://example')
        client.collection = collection

        def client_factory(*args, **kwargs):
            created.append(client)
            return client

        fake_pymongo = types.ModuleType('pymongo')
        fake_pymongo.MongoClient = client_factory
        monkeypatch.setitem(sys.modules, 'pymongo', fake_pymongo)
        return MongoFaceIdentityStore('mongodb://example', model_key=model_key,
                                      robot_id=robot_id), collection

    return make


def make_identity(unique_id='U1'):
    identity = FaceIdentityCluster(unique_id=unique_id, creation_timestamp=1.0,
                                   last_seen_timestamp=2.0)
    identity.mean_embedding = normalize_embedding(np.ones(DIM, dtype=np.float32))
    identity.all_embeddings = [identity.mean_embedding.copy()]
    identity.confirmed = True
    return identity


def test_model_key_is_the_profile_scope(store_factory):
    store, collection = store_factory('vggface2-aligned')

    assert store.model_key == 'vggface2-aligned'
    # A gallery can be found again by scope without a server round trip, and two scopes
    # can never mint the same U<n>.
    assert ([('model_key', 1), ('unique_id', 1)], {'unique': True}) in collection.indexes


def test_save_scopes_the_document_and_records_provenance(store_factory):
    store, collection = store_factory('vggface2-aligned', robot_id='robot_a')

    store.save(make_identity())

    query, update, upsert = collection.updates[-1]
    assert query == {'model_key': 'vggface2-aligned', 'unique_id': 'U1'}
    assert update['$set']['model_key'] == 'vggface2-aligned'
    assert update['$set']['unique_id'] == 'U1'
    assert update['$set']['updated_by_robot'] == 'robot_a'
    assert update['$set']['last_seen_by_robot'] == 'robot_a'
    assert update['$setOnInsert']['created_by_robot'] == 'robot_a'
    assert 'created_at' in update['$setOnInsert']
    assert upsert is True


def test_revision_is_incremented_atomically(store_factory):
    store, collection = store_factory('vggface2-aligned', robot_id='robot_a')

    store.save(make_identity())
    store.save(make_identity())

    # $inc is applied server-side, so two robots writing the same profile cannot both
    # read the same revision and lose one of the updates.
    for _, update, _ in collection.updates:
        assert update['$inc'] == {'revision': 1}
        assert 'revision' not in update['$set']


def test_single_robot_mode_leaves_provenance_empty(store_factory):
    store, collection = store_factory('vggface2-aligned')

    store.save(make_identity())

    _, update, _ = collection.updates[-1]
    assert update['$set']['updated_by_robot'] == ''
    assert update['$setOnInsert']['created_by_robot'] == ''


def test_save_without_embedding_writes_nothing(store_factory):
    store, collection = store_factory('vggface2-aligned')

    store.save(FaceIdentityCluster(unique_id='U2', creation_timestamp=1.0,
                                   last_seen_timestamp=2.0))

    assert collection.updates == []


def test_updated_at_is_indexed_for_incremental_refresh(store_factory):
    store, collection = store_factory('vggface2-aligned')

    assert ('updated_at', {}) in collection.indexes


def test_delete_is_scoped(store_factory):
    store, collection = store_factory('vggface2-aligned')

    store.delete('U7')

    # Deleting by id alone would let one robot remove another scope's profile.
    assert collection.deletes[-1] == {'model_key': 'vggface2-aligned', 'unique_id': 'U7'}
