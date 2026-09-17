"""Persistent face identity management.

Embeddings arrive keyed by a transient face track id (``face_3``); this module decides
which persistent identity (``U1``, ``U2``...) each one belongs to. It owns the identity
population, the matching rules, merging, cleanup and incremental MongoDB persistence.

The rules are ported from the speech diarization identity layer
(``eut_speech_audio_processing/.../voice_identity_manager.py``) and calibrated on
``video_3.mp4`` with ``tools/evaluate_identity_manager.py``:

* a new identity is seeded from several consistent, good-quality samples of one track,
  never from a single embedding (single same-person pairs score as low as 0.25);
* matching needs an absolute score **and** a best-vs-second margin, with a relaxed bar
  for young (tentative) identities whose mean is still noisy;
* assignment inside a frame is exclusive, so two faces never share an identity;
* a track keeps its identity unless another one is clearly better, but never below
  the young threshold;
* only good-quality faces (near-frontal, confidently detected) teach an identity;
  poor faces may be matched to a known identity but otherwise stay unlabeled;
* each identity keeps a *diverse* gallery (near-duplicate frames are skipped and the most
  redundant sample is evicted), so it remembers every look instead of the last seconds;
* scores combine the gallery mean with the closest gallery samples;
* a seed that clearly matches an existing identity joins it instead of creating a new one;
* fragments are merged, and young strays are absorbed, unless the two identities were
  ever seen at the same time on different faces (then they are different people);
* confirmed identities are written to MongoDB incrementally and throttled, not only at
  shutdown, and a database outage never stops recognition.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set

import numpy as np

STATUS_UNKNOWN = 0
STATUS_TENTATIVE = 1
STATUS_CONFIRMED = 2


@dataclass
class FaceIdentityCluster:
    """One persistent identity: its embedding population plus statistics."""

    unique_id: str
    creation_timestamp: float
    last_seen_timestamp: float

    all_embeddings: List[np.ndarray] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None

    current_track_id: Optional[str] = None
    total_detections: int = 0
    good_samples: int = 0
    first_learned_timestamp: Optional[float] = None
    last_learned_timestamp: Optional[float] = None
    confirmed: bool = False
    quality_score: float = 0.0

    custom_name: Optional[str] = None
    # Identities seen in the same frame on another face: never the same person
    co_occurring: Set[str] = field(default_factory=set)
    unsaved_updates: int = 0
    persisted: bool = False
    last_saved_timestamp: float = 0.0
    # Cached np.stack(all_embeddings); rebuilt when the gallery changes
    gallery_matrix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


class FaceAssignment(NamedTuple):
    unique_id: Optional[str]
    confidence: float
    status: int


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Face embedding must be a finite, non-zero vector")
    return vector / norm


class MongoFaceIdentityStore:
    """MongoDB persistence for :class:`FaceIdentityCluster`.

    Documents are namespaced by ``model_key`` so embeddings of different models are
    never compared. Documents without ``model_key`` (pre-refactor format) are ignored.
    """

    def __init__(self, mongo_uri: str, model_key: str, database_name: str = "face_recognition_db",
                 collection_name: str = "identity_database", save_last_n_embeddings: int = 100) -> None:
        from pymongo import MongoClient

        self._model_key = model_key
        self._save_last_n = max(1, save_last_n_embeddings)
        self._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self._client.admin.command("ping")
        self._collection = self._client[database_name][collection_name]
        self._collection.create_index([("model_key", 1), ("unique_id", 1)], unique=True)

    def count_legacy_documents(self) -> int:
        return self._collection.count_documents({"model_key": {"$exists": False}})

    def count_other_model_documents(self) -> int:
        return self._collection.count_documents({"model_key": {"$exists": True, "$ne": self._model_key}})

    def max_user_number(self) -> int:
        """Highest U<n> ever stored under any model key, so new identities never reuse an id
        that downstream consumers (e.g. PersonManager) may still link to another face."""
        highest = 0
        for document in self._collection.find({}, {"unique_id": 1}):
            match = re.search(r"(\d+)$", str(document.get("unique_id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return highest

    def load(self) -> List[FaceIdentityCluster]:
        identities = []
        for document in self._collection.find({"model_key": self._model_key}):
            embeddings = [normalize_embedding(np.asarray(e, dtype=np.float32))
                          for e in document.get("embeddings", [])]
            mean = document.get("mean_embedding")
            if mean is None and not embeddings:
                continue
            mean_vector = (normalize_embedding(np.asarray(mean, dtype=np.float32)) if mean is not None
                           else normalize_embedding(np.mean(np.stack(embeddings), axis=0)))
            identities.append(FaceIdentityCluster(
                unique_id=document["unique_id"],
                creation_timestamp=float(document.get("creation_timestamp", 0.0)),
                last_seen_timestamp=float(document.get("last_seen_timestamp", 0.0)),
                all_embeddings=embeddings or [mean_vector.copy()],
                mean_embedding=mean_vector,
                total_detections=int(document.get("total_detections", 0)),
                good_samples=int(document.get("good_samples", len(embeddings))),
                confirmed=bool(document.get("confirmed", True)),
                quality_score=float(document.get("quality_score", 0.0)),
                custom_name=document.get("custom_name"),
                co_occurring=set(document.get("co_occurring", [])),
            ))
        return identities

    def save(self, identity: FaceIdentityCluster) -> None:
        if identity.mean_embedding is None:
            return
        self._collection.update_one(
            {"model_key": self._model_key, "unique_id": identity.unique_id},
            {"$set": {
                "model_key": self._model_key,
                "unique_id": identity.unique_id,
                "creation_timestamp": float(identity.creation_timestamp),
                "last_seen_timestamp": float(identity.last_seen_timestamp),
                "total_detections": int(identity.total_detections),
                "good_samples": int(identity.good_samples),
                "confirmed": bool(identity.confirmed),
                "quality_score": float(identity.quality_score),
                "custom_name": identity.custom_name,
                "co_occurring": sorted(identity.co_occurring),
                "embeddings": [e.astype(float).tolist() for e in identity.all_embeddings[-self._save_last_n:]],
                "mean_embedding": identity.mean_embedding.astype(float).tolist(),
                "updated_at": time.time(),
            }},
            upsert=True,
        )

    def delete(self, unique_id: str) -> None:
        self._collection.delete_one({"model_key": self._model_key, "unique_id": unique_id})

    def close(self) -> None:
        self._client.close()


class FaceIdentityManager:
    """Assign transient face tracks to persistent identities."""

    def __init__(
        self,
        *,
        logger=None,
        store=None,
        clock: Callable[[], float] = time.time,
        similarity_threshold: float = 0.50,
        young_identity_threshold: float = 0.40,
        match_margin: float = 0.08,
        stickiness_margin: float = 0.20,
        merge_threshold: float = 0.50,
        seed_match_threshold: float = 0.50,
        min_learn_quality: float = 0.50,
        min_seed_samples: int = 4,
        seed_consistency: float = 0.55,
        seed_pairwise_consistency: float = 0.40,
        min_confirm_samples: int = 12,
        min_confirm_seconds: float = 3.0,
        max_embeddings_per_identity: int = 100,
        redundancy_threshold: float = 0.92,
        gallery_top_k: int = 3,
        merge_check_interval: float = 1.0,
        identity_timeout: float = 30.0,
        track_timeout: float = 2.0,
        persist_every: int = 20,
        min_persist_interval: float = 10.0,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._store = store
        self._clock = clock
        self.similarity_threshold = similarity_threshold
        self.young_identity_threshold = min(young_identity_threshold, similarity_threshold)
        self.match_margin = match_margin
        self.stickiness_margin = stickiness_margin
        self.merge_threshold = merge_threshold
        self.seed_match_threshold = seed_match_threshold
        self.min_learn_quality = min_learn_quality
        self.min_seed_samples = max(1, min_seed_samples)
        self.seed_consistency = seed_consistency
        self.seed_pairwise_consistency = seed_pairwise_consistency
        self.min_confirm_samples = min_confirm_samples
        self.min_confirm_seconds = min_confirm_seconds
        self.max_embeddings_per_identity = max(2, max_embeddings_per_identity)
        self.redundancy_threshold = redundancy_threshold
        self.gallery_top_k = max(1, gallery_top_k)
        self.merge_check_interval = merge_check_interval
        self.identity_timeout = identity_timeout
        self.track_timeout = track_timeout
        self.persist_every = max(1, persist_every)
        self.min_persist_interval = min_persist_interval

        self.identity_clusters: Dict[str, FaceIdentityCluster] = {}
        self.track_id_to_unique_id: Dict[str, str] = {}
        self._track_last_seen: Dict[str, float] = {}
        self._seed_buffers: Dict[str, List[np.ndarray]] = {}
        self._next_user_number = 1
        self._last_merge_check = float("-inf")

        self.total_identities_created = 0
        self.total_seed_rematches = 0
        self.total_identity_merges = 0
        self.total_saves = 0

        if self._store is not None:
            if hasattr(self._store, "max_user_number"):
                self._next_user_number = self._store.max_user_number() + 1
            for identity in self._store.load():
                identity.persisted = True
                identity.last_saved_timestamp = self._clock()
                self.identity_clusters[identity.unique_id] = identity
                self._next_user_number = max(self._next_user_number, self._user_number(identity.unique_id) + 1)
        loaded = ", ".join(sorted(self.identity_clusters, key=self._user_number))
        self._logger.info(f"Face identity manager ready with {len(self.identity_clusters)} persistent identities"
                          + (f": {loaded}" if loaded else ""))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_new_embedding_batch(self, track_embeddings: Dict[str, np.ndarray],
                                    quality: Dict[str, float] | float = 1.0) -> Dict[str, FaceAssignment]:
        """Assign every face of one frame. Unmatched faces get ``unique_id=None``."""
        if not track_embeddings:
            return {}
        now = self._clock()

        if now - self._last_merge_check >= self.merge_check_interval:
            self._last_merge_check = now
            self._merge_similar_identities()
            self._absorb_stray_identities()

        vectors: Dict[str, np.ndarray] = {}
        for track_id, embedding in track_embeddings.items():
            try:
                vectors[track_id] = normalize_embedding(embedding)
            except ValueError:
                continue
        track_ids = list(vectors)
        assignments = self._assign_batch(track_ids, vectors)
        claimed = {uid for uid, _ in assignments.values() if uid is not None}

        results: Dict[str, FaceAssignment] = {}
        for track_id in track_ids:
            self._track_last_seen[track_id] = now
            unique_id, score = assignments[track_id]
            good = self._per_track(quality, track_id) >= self.min_learn_quality

            if unique_id is None:
                if self.track_id_to_unique_id.get(track_id) not in self.identity_clusters:
                    self.track_id_to_unique_id.pop(track_id, None)
                seeded = self._seed(track_id, vectors[track_id], now, claimed) if good else None
                if seeded is None:
                    results[track_id] = FaceAssignment(None, 0.0, STATUS_UNKNOWN)
                    continue
                unique_id, score = seeded
                claimed.add(unique_id)
                identity = self.identity_clusters[unique_id]
                results[track_id] = FaceAssignment(
                    unique_id, score, STATUS_CONFIRMED if identity.confirmed else STATUS_TENTATIVE)
                continue

            self._seed_buffers.pop(track_id, None)
            identity = self.identity_clusters[unique_id]
            self.track_id_to_unique_id[track_id] = unique_id
            identity.current_track_id = track_id
            identity.last_seen_timestamp = now
            identity.total_detections += 1
            bar = self.similarity_threshold if identity.confirmed else self.young_identity_threshold
            if good and score >= bar:
                self._add_embedding(identity, vectors[track_id], now)
            results[track_id] = FaceAssignment(
                unique_id, max(0.0, score), STATUS_CONFIRMED if identity.confirmed else STATUS_TENTATIVE)

        self._record_co_occurrence({a.unique_id for a in results.values() if a.unique_id})
        self._cleanup_stale_tracks(now)
        self.cleanup_inactive_identities()
        return results

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _gallery(self, identity: FaceIdentityCluster) -> np.ndarray:
        if identity.gallery_matrix is None or len(identity.gallery_matrix) != len(identity.all_embeddings):
            identity.gallery_matrix = np.stack(identity.all_embeddings)
        return identity.gallery_matrix

    def _score_matrix(self, queries: np.ndarray, identity_ids: Sequence[str]) -> np.ndarray:
        """Similarity of each query to each identity: half gallery mean, half its closest samples."""
        scores = np.empty((len(queries), len(identity_ids)), dtype=np.float32)
        for column, unique_id in enumerate(identity_ids):
            identity = self.identity_clusters[unique_id]
            gallery = self._gallery(identity)
            sample_scores = queries @ gallery.T
            k = min(self.gallery_top_k, gallery.shape[0])
            closest = np.partition(sample_scores, -k, axis=1)[:, -k:].mean(axis=1)
            scores[:, column] = 0.5 * (queries @ identity.mean_embedding) + 0.5 * closest
        return scores

    def _pair_score(self, first: FaceIdentityCluster, second: FaceIdentityCluster, top_k: int = 5) -> float:
        cross = (self._gallery(first) @ self._gallery(second).T).ravel()
        k = min(top_k, cross.size)
        closest = float(np.partition(cross, -k)[-k:].mean())
        return 0.5 * float(first.mean_embedding @ second.mean_embedding) + 0.5 * closest

    def _assign_batch(self, track_ids: Sequence[str], vectors: Dict[str, np.ndarray]):
        result = {track_id: (None, 0.0) for track_id in track_ids}
        identity_ids = [uid for uid, c in self.identity_clusters.items() if c.mean_embedding is not None]
        if not identity_ids or not track_ids:
            return result

        similarity = self._score_matrix(np.stack([vectors[t] for t in track_ids]), identity_ids)

        # Most confident faces first, so a strong match claims its identity before an ambiguous one.
        order = sorted(range(len(track_ids)), key=lambda i: float(np.max(similarity[i])), reverse=True)
        claimed: Set[int] = set()
        for row in order:
            track_id = track_ids[row]
            scores = similarity[row]
            available = [j for j in range(len(identity_ids)) if j not in claimed]
            if not available:
                break
            ranked = sorted(available, key=lambda j: float(scores[j]), reverse=True)
            best = ranked[0]
            best_score = float(scores[best])
            second_score = float(scores[ranked[1]]) if len(ranked) > 1 else -1.0

            chosen = None
            previous = self.track_id_to_unique_id.get(track_id)
            if previous in identity_ids:
                previous_index = identity_ids.index(previous)
                if previous_index not in claimed:
                    previous_score = float(scores[previous_index])
                    if (previous_score >= self.young_identity_threshold
                            and previous_score >= best_score - self.stickiness_margin):
                        chosen, best_score = previous_index, previous_score

            if chosen is None:
                required = self._required_score(self.identity_clusters[identity_ids[best]])
                # A near-tie means the faces cannot be told apart: leave unlabeled.
                if best_score >= required and (best_score - second_score >= self.match_margin or len(ranked) == 1):
                    chosen = best

            if chosen is not None:
                claimed.add(chosen)
                result[track_id] = (identity_ids[chosen], best_score)
        return result

    def _required_score(self, identity: FaceIdentityCluster) -> float:
        return self.similarity_threshold if identity.confirmed else self.young_identity_threshold

    def score(self, unique_id: str, embedding: np.ndarray) -> Optional[float]:
        identity = self.identity_clusters.get(unique_id)
        if identity is None or identity.mean_embedding is None:
            return None
        return float(self._score_matrix(normalize_embedding(embedding)[None, :], [unique_id])[0, 0])

    # ------------------------------------------------------------------
    # Identity lifecycle
    # ------------------------------------------------------------------

    def _seed(self, track_id: str, vector: np.ndarray, now: float,
              claimed: Set[str]) -> Optional[Tuple[str, float]]:
        """Collect good samples of an unmatched track. Once they agree, join the existing
        identity they clearly match, or create a new one. Returns (unique_id, score)."""
        buffer = self._seed_buffers.setdefault(track_id, [])
        buffer.append(vector)
        del buffer[:-self.min_seed_samples * 2]
        if len(buffer) < self.min_seed_samples:
            return None
        recent = np.stack(buffer[-self.min_seed_samples:])
        mean = normalize_embedding(recent.mean(axis=0))
        consistency = float(np.min(recent @ mean))
        pairwise = recent @ recent.T
        pairwise_mean = (float((pairwise.sum() - np.trace(pairwise)) / (len(recent) * (len(recent) - 1)))
                         if len(recent) > 1 else 1.0)
        # Both checks: a track alternating between two people has a mean halfway
        # between them, so every sample still scores ~0.7 against it; the pairwise
        # mean exposes the mixture.
        if consistency < self.seed_consistency or pairwise_mean < self.seed_pairwise_consistency:
            return None  # the track mixes people or poses: keep waiting

        # The averaged seed is far less noisy than single frames: if it clearly matches an
        # identity not visible elsewhere in this frame, this is a known person in a new look.
        candidates = [uid for uid, c in self.identity_clusters.items()
                      if c.mean_embedding is not None and uid not in claimed]
        if candidates:
            scores = self._score_matrix(mean[None, :], candidates)[0]
            order = np.argsort(scores)[::-1]
            best_score = float(scores[order[0]])
            second_score = float(scores[order[1]]) if len(order) > 1 else -1.0
            if best_score >= self.seed_match_threshold and best_score - second_score >= self.match_margin:
                unique_id = candidates[int(order[0])]
                identity = self.identity_clusters[unique_id]
                for sample in recent:
                    self._add_embedding(identity, sample, now)
                identity.last_seen_timestamp = now
                identity.current_track_id = track_id
                identity.total_detections += len(recent)
                self.track_id_to_unique_id[track_id] = unique_id
                self._seed_buffers.pop(track_id, None)
                self.total_seed_rematches += 1
                self._logger.info(f"Track {track_id} joined face identity {unique_id} "
                                  f"(seed score {best_score:.3f}, second {second_score:.3f})")
                return unique_id, best_score

        unique_id = f"U{self._next_user_number}"
        self._next_user_number += 1
        identity = FaceIdentityCluster(unique_id=unique_id, creation_timestamp=now, last_seen_timestamp=now,
                                       current_track_id=track_id, total_detections=len(recent))
        self.identity_clusters[unique_id] = identity
        for sample in recent:
            self._add_embedding(identity, sample, now)
        self.track_id_to_unique_id[track_id] = unique_id
        self._seed_buffers.pop(track_id, None)
        self.total_identities_created += 1
        self._logger.info(f"New face identity {unique_id} (track {track_id}, seed consistency {consistency:.3f})")
        return unique_id, 1.0

    def _add_embedding(self, identity: FaceIdentityCluster, vector: np.ndarray, now: float) -> None:
        """Count a good observation; store it only if it adds a look the gallery lacks."""
        identity.good_samples += 1
        if identity.first_learned_timestamp is None:
            identity.first_learned_timestamp = now
        identity.last_learned_timestamp = now

        if self._store_sample(identity, vector):
            identity.unsaved_updates += 1

        if not identity.confirmed and (
                identity.good_samples >= self.min_confirm_samples
                and now - identity.first_learned_timestamp >= self.min_confirm_seconds):
            identity.confirmed = True
            identity.unsaved_updates += 1
            self._logger.info(f"Face identity {identity.unique_id} confirmed ({identity.good_samples} samples, "
                              f"{len(identity.all_embeddings)} distinct)")
        self._maybe_save(identity, now)

    def _store_sample(self, identity: FaceIdentityCluster, vector: np.ndarray) -> bool:
        """Add a sample to the gallery unless it is a near-duplicate. Returns True if stored."""
        if identity.all_embeddings and float(np.max(self._gallery(identity) @ vector)) >= self.redundancy_threshold:
            return False  # near-duplicate of a stored look (e.g. a static face)
        identity.all_embeddings.append(vector.copy())
        if len(identity.all_embeddings) > self.max_embeddings_per_identity:
            # Evict the most redundant sample, not the oldest: keep every distinct look.
            gallery = np.stack(identity.all_embeddings)
            similarity = gallery @ gallery.T
            np.fill_diagonal(similarity, -1.0)
            del identity.all_embeddings[int(np.argmax(similarity.max(axis=1)))]
        identity.gallery_matrix = None
        identity.mean_embedding = normalize_embedding(np.mean(self._gallery(identity), axis=0))
        identity.quality_score = self._quality_score(identity)
        return True

    def _quality_score(self, identity: FaceIdentityCluster) -> float:
        if len(identity.all_embeddings) < 2 or identity.mean_embedding is None:
            return 0.0
        consistency = float(np.mean(self._gallery(identity) @ identity.mean_embedding))
        population = min(len(identity.all_embeddings) / float(self.max_embeddings_per_identity), 1.0)
        return 0.7 * consistency + 0.3 * population

    def _record_co_occurrence(self, unique_ids: Set[str]) -> None:
        for first, second in combinations(sorted(unique_ids), 2):
            a, b = self.identity_clusters.get(first), self.identity_clusters.get(second)
            if a is None or b is None or second in a.co_occurring:
                continue
            a.co_occurring.add(second)
            b.co_occurring.add(first)
            a.unsaved_updates += 1
            b.unsaved_updates += 1

    # ------------------------------------------------------------------
    # Merging and cleanup
    # ------------------------------------------------------------------

    def _merge_similar_identities(self) -> None:
        """Fold fragments of one person together. Identities ever seen together are never merged."""
        candidates = [uid for uid, c in self.identity_clusters.items()
                      if c.mean_embedding is not None and c.good_samples >= self.min_confirm_samples]
        pairs = []
        for first, second in combinations(candidates, 2):
            a, b = self.identity_clusters[first], self.identity_clusters[second]
            if second in a.co_occurring:
                continue
            score = self._pair_score(a, b)
            if score >= self.merge_threshold:
                pairs.append((score, first, second))
        merged: Set[str] = set()
        for score, first, second in sorted(pairs, reverse=True):
            if first in merged or second in merged:
                continue
            keep, drop = sorted((first, second), key=self._user_number)
            if self.merge_identities(keep, drop):
                merged.add(drop)
                self._logger.info(f"Merged face identity {drop} into {keep} (score {score:.3f})")

    def _absorb_stray_identities(self) -> None:
        """Fold young identities into a confirmed one they clearly belong to."""
        mature = [uid for uid, c in self.identity_clusters.items() if c.confirmed and c.mean_embedding is not None]
        young = [uid for uid, c in self.identity_clusters.items() if not c.confirmed and c.mean_embedding is not None]
        for uid in young:
            stray = self.identity_clusters.get(uid)
            options = [m for m in mature if m in self.identity_clusters and m not in stray.co_occurring]
            if stray is None or not options:
                continue
            scores = sorted(((self._pair_score(stray, self.identity_clusters[m]), m) for m in options), reverse=True)
            best, keep = scores[0]
            second = scores[1][0] if len(scores) > 1 else -1.0
            if best < self.similarity_threshold or best - second < self.match_margin:
                continue
            if self.merge_identities(keep, uid):
                self._logger.info(f"Absorbed stray face identity {uid} into {keep} (score {best:.3f})")

    def merge_identities(self, keep_id: str, drop_id: str) -> bool:
        if keep_id == drop_id or keep_id not in self.identity_clusters or drop_id not in self.identity_clusters:
            return False
        keep = self.identity_clusters[keep_id]
        drop = self.identity_clusters.pop(drop_id)
        now = self._clock()
        for sample in drop.all_embeddings:
            self._store_sample(keep, sample)
        keep.total_detections += drop.total_detections
        keep.good_samples += drop.good_samples
        keep.creation_timestamp = min(keep.creation_timestamp, drop.creation_timestamp)
        keep.last_seen_timestamp = max(keep.last_seen_timestamp, drop.last_seen_timestamp)
        firsts = [t for t in (keep.first_learned_timestamp, drop.first_learned_timestamp) if t is not None]
        keep.first_learned_timestamp = min(firsts) if firsts else None
        keep.confirmed = keep.confirmed or drop.confirmed
        keep.custom_name = keep.custom_name or drop.custom_name
        keep.co_occurring |= drop.co_occurring - {keep_id}
        for other in self.identity_clusters.values():
            if drop_id in other.co_occurring:
                other.co_occurring.discard(drop_id)
                other.co_occurring.add(keep_id)
        keep.unsaved_updates += 1
        for track_id, unique_id in list(self.track_id_to_unique_id.items()):
            if unique_id == drop_id:
                self.track_id_to_unique_id[track_id] = keep_id
        if self._store is not None and drop.persisted:
            try:
                self._store.delete(drop_id)
            except Exception as error:
                self._logger.warning(f"Could not delete merged face identity {drop_id}: {error}")
        self._save(keep, now)
        self.total_identity_merges += 1
        return True

    def cleanup_inactive_track_mappings(self, active_track_ids: Set[str]) -> None:
        """Forget tracks that are gone, so a reused track id never inherits an identity."""
        for track_id in [t for t in self.track_id_to_unique_id if t not in active_track_ids]:
            self._forget_track(track_id)
        for track_id in [t for t in self._seed_buffers if t not in active_track_ids]:
            self._seed_buffers.pop(track_id, None)

    def _cleanup_stale_tracks(self, now: float) -> None:
        for track_id in [t for t, seen in self._track_last_seen.items() if now - seen > self.track_timeout]:
            self._forget_track(track_id)
            self._track_last_seen.pop(track_id, None)

    def _forget_track(self, track_id: str) -> None:
        unique_id = self.track_id_to_unique_id.pop(track_id, None)
        self._seed_buffers.pop(track_id, None)
        identity = self.identity_clusters.get(unique_id) if unique_id else None
        if identity is not None and identity.current_track_id == track_id:
            identity.current_track_id = None

    def cleanup_inactive_identities(self) -> None:
        """Drop tentative identities that went quiet. Confirmed or persisted ones are kept."""
        now = self._clock()
        for unique_id, identity in list(self.identity_clusters.items()):
            if identity.confirmed or identity.persisted:
                continue
            if now - identity.last_seen_timestamp > self.identity_timeout:
                del self.identity_clusters[unique_id]
                for track_id, mapped in list(self.track_id_to_unique_id.items()):
                    if mapped == unique_id:
                        del self.track_id_to_unique_id[track_id]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _maybe_save(self, identity: FaceIdentityCluster, now: float) -> None:
        if not identity.confirmed:
            return
        if not identity.persisted or (identity.unsaved_updates >= self.persist_every
                                      and now - identity.last_saved_timestamp >= self.min_persist_interval):
            self._save(identity, now)

    def _save(self, identity: FaceIdentityCluster, now: float) -> None:
        if self._store is None or not identity.confirmed:
            return
        try:
            self._store.save(identity)
        except Exception as error:  # a database outage must never stop recognition
            self._logger.warning(f"Could not persist face identity {identity.unique_id}: {error}")
            return
        if not identity.persisted:
            self._logger.info(f"Persisted face identity {identity.unique_id} ({len(identity.all_embeddings)} embeddings)")
        identity.persisted = True
        identity.unsaved_updates = 0
        identity.last_saved_timestamp = now
        self.total_saves += 1

    def flush(self, force: bool = False) -> None:
        """Save confirmed identities with pending updates (throttled unless ``force``)."""
        now = self._clock()
        for identity in list(self.identity_clusters.values()):
            if not identity.confirmed or (identity.persisted and identity.unsaved_updates == 0):
                continue
            if force or not identity.persisted or now - identity.last_saved_timestamp >= self.min_persist_interval:
                self._save(identity, now)

    def close(self) -> None:
        self.flush(force=True)
        if self._store is not None:
            self._store.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_unique_id_for_track(self, track_id: str) -> Optional[str]:
        return self.track_id_to_unique_id.get(track_id)

    def get_identity_info(self, unique_id: str) -> Optional[Dict]:
        identity = self.identity_clusters.get(unique_id)
        if identity is None:
            return None
        return {
            "unique_id": unique_id,
            "creation_timestamp": identity.creation_timestamp,
            "last_seen_timestamp": identity.last_seen_timestamp,
            "total_detections": identity.total_detections,
            "good_samples": identity.good_samples,
            "confirmed": identity.confirmed,
            "persisted": identity.persisted,
            "quality_score": identity.quality_score,
            "num_embeddings": len(identity.all_embeddings),
            "current_track_id": identity.current_track_id,
            "custom_name": identity.custom_name,
        }

    def get_all_identities(self) -> Dict[str, Dict]:
        return {uid: self.get_identity_info(uid) for uid in self.identity_clusters}

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_identities": len(self.identity_clusters),
            "confirmed_identities": sum(c.confirmed for c in self.identity_clusters.values()),
            "total_identities_created": self.total_identities_created,
            "total_seed_rematches": self.total_seed_rematches,
            "total_identity_merges": self.total_identity_merges,
            "total_saves": self.total_saves,
            "active_tracks": len(self.track_id_to_unique_id),
        }

    def set_custom_name(self, unique_id: str, custom_name: str) -> bool:
        identity = self.identity_clusters.get(unique_id)
        if identity is None:
            return False
        identity.custom_name = custom_name
        identity.unsaved_updates += 1
        self._save(identity, self._clock())
        return True

    @staticmethod
    def _per_track(value: Dict[str, float] | float, track_id: str) -> float:
        if isinstance(value, dict):
            return float(value.get(track_id, 0.0))
        return float(value)

    @staticmethod
    def _user_number(unique_id: str) -> int:
        match = re.search(r"(\d+)$", unique_id)
        return int(match.group(1)) if match else 0
