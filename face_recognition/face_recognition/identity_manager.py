"""
Identity Management System for Face Recognition Package

This system manages persistent identity tracking across changing track IDs by:
1. Assigning unique persistent IDs to detected humans
2. Using face embedding clustering for re-identification
3. Maintaining identity history and statistics
4. Handling identity merging when multiple track IDs belong to same person

Adapted from EUT YOLO core identity management system.
"""

import time
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    from sklearn.cluster import DBSCAN
    from scipy.spatial.distance import cosine
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@dataclass
class IdentityCluster:
    """
    Represents a unique identity with all associated data.
    """
    unique_id: str  # U1, U2, or custom name like "John"
    creation_timestamp: float
    last_seen_timestamp: float
    
    # Embedding data
    all_embeddings: List[np.ndarray] = field(default_factory=list)
    embedding_confidences: List[float] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None
    
    # Track ID associations
    associated_track_ids: Set[int] = field(default_factory=set)
    current_track_id: int = field(default_factory=int)  # Currently active
    
    # Statistics
    total_detections: int = 0
    quality_score: float = 0.0  # Based on embedding consistency and detection count
    
    # User-defined properties
    custom_name: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class IdentityManager:
    """
    Manages persistent identity tracking across changing track IDs.
    Adapted from EUT YOLO identity management system.
    """
    
    def __init__(self, 
                 max_embeddings_per_identity: int = 50,
                 similarity_threshold: float = 0.6,
                 track_identity_stickiness_margin: float = 0.4,
                 clustering_threshold: float = 0.7,
                 embedding_inclusion_threshold: float = 0.6,
                 identity_timeout: float = 60.0,
                 min_detections_for_stable_identity: int = 5,
                 enable_debug_prints: bool = False,
                 identity_database_path: Optional[str] = None,
                 use_ewma_for_mean: bool = False,
                 ewma_alpha: float = 0.6):
        """
        Initialize the identity manager.
        
        Args:
            max_embeddings_per_identity: Maximum embeddings to store per identity
            similarity_threshold: Threshold for considering embeddings similar (minimum for identity assignment)
            track_identity_stickiness_margin: Maximum allowed similarity difference to prefer the previously assigned identity for a track
            clustering_threshold: Threshold for clustering embeddings into identities
            embedding_inclusion_threshold: Threshold for including embeddings in identity cluster (must be >= similarity_threshold)
            identity_timeout: Time (seconds) after which inactive identity is considered lost
            min_detections_for_stable_identity: Minimum detections needed for stable identity
            enable_debug_prints: Enable detailed debug prints for embedding similarities and clustering
            identity_database_path: Path to JSON file for persistent identity storage (optional)
            use_ewma_for_mean: Whether to use Exponentially Weighted Moving Average for updating mean embeddings
            ewma_alpha: Learning rate for EWMA (0 < alpha < 1). Higher values adapt faster to new embeddings.
        """
        self.max_embeddings_per_identity = max_embeddings_per_identity
        self.similarity_threshold = similarity_threshold
        self.track_identity_stickiness_margin = track_identity_stickiness_margin
        self.clustering_threshold = clustering_threshold
        self.embedding_inclusion_threshold = max(embedding_inclusion_threshold, similarity_threshold)
        self.identity_timeout = identity_timeout
        self.min_detections_for_stable_identity = min_detections_for_stable_identity
        self.enable_debug_prints = enable_debug_prints
        self.identity_database_path = identity_database_path
        
        # EWMA parameters for mean embedding updates
        self.use_ewma_for_mean = use_ewma_for_mean
        self.ewma_alpha = max(0.01, min(0.99, ewma_alpha))  # Clamp alpha to (0.01, 0.99) for stability
        
        # Core data structures
        self.identity_clusters: Dict[str, IdentityCluster] = {}  # unique_id -> IdentityCluster
        self.track_id_to_unique_id: Dict[int, str] = {}  # track_id -> unique_id
        
        # Identity creation tracking
        self.next_user_number = 1
        self.pending_identities: Dict[int, List[np.ndarray]] = {}  # track_id -> embeddings
        
        # Performance tracking
        self.total_identities_created = 0
        self.total_re_identifications = 0
        self.total_identity_merges = 0
        
        # Fixed color mapping for consistent visualization
        self.identity_color_mapping: Dict[str, str] = {}  # unique_id -> color
        self.available_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
                                'magenta', 'yellow', 'navy', 'lime', 'maroon', 'teal', 'silver', 'gold', 'indigo', 'coral']
        
        # Load existing identities from database if specified
        if self.identity_database_path:
            self.load_identity_database()

    def process_new_embedding_batch(self, track_embeddings: Dict[int, np.ndarray]) -> Dict[int, Tuple[Optional[str], float]]:
        """
        Process multiple new face embeddings in batch and assign/update identities.
        
        This batched version allows for optimal identity assignment by comparing all
        embeddings against all identities simultaneously, preventing conflicts and
        enabling better assignment decisions.
        
        Args:
            track_embeddings: Dictionary mapping track_id -> embedding
            
        Returns:
            Dictionary mapping track_id -> (unique_id, confidence_score)
            where confidence_score is the cosine similarity of the best match
        """
        current_time = time.time()
        results = {}
        
        if not track_embeddings:
            return results

        # Perform merges before matching
        self._check_and_perform_merges_batch()

        # Normalize all embeddings
        normalized_embeddings = {}
        for track_id, embedding in track_embeddings.items():
            normalized_embeddings[track_id] = self._normalize_embedding(embedding)
        
        if self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] Processing {len(normalized_embeddings)} embeddings in batch")

        # Get batched identity matches with similarity matrix
        match_results = self._find_best_identity_match_batch(list(normalized_embeddings.values()), list(normalized_embeddings.keys()))
        
        # match_results contains: (similarity_matrix, best_matches)
        similarity_matrix, best_matches = match_results
        track_ids = list(normalized_embeddings.keys())

        # Process each track's assignment
        for i, track_id in enumerate(track_ids):
            unique_id, confidence = best_matches[i]
            
            if unique_id is not None:
                # Update existing identity
                self._update_existing_identity(unique_id, track_id, normalized_embeddings[track_id], confidence, current_time, confidence)
                results[track_id] = (unique_id, confidence)
                if self.enable_debug_prints:
                    print(f"[IDENTITY_DEBUG] Track {track_id} assigned to existing identity {unique_id} with confidence {confidence:.3f}")
            else:
                # Create new identity
                new_unique_id = self._create_new_identity(track_id, [normalized_embeddings[track_id]], confidence, current_time)
                results[track_id] = (new_unique_id, confidence)
                if self.enable_debug_prints:
                    print(f"[IDENTITY_DEBUG] Track {track_id} assigned to new identity {new_unique_id}")

        if self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] Batch processing complete. Total identities: {len(self.identity_clusters)}")
             
        return results

    def _find_best_identity_match_batch(self, embeddings: List[np.ndarray], track_ids: List[int] = None, 
                                       mode: str = 'accurate', n_recent_embeddings: int = 20, n_top_embed: int = 10) -> Tuple[np.ndarray, List[Tuple[Optional[str], float]]]:
        """
        Find the best matching existing identities for a batch of embeddings with exclusive assignment.
        Each identity can only be assigned to one track (1:1 mapping).
        
        Args:
            embeddings: List of normalized embeddings to match
            track_ids: List of track IDs for debugging output (optional)
            n_recent_embeddings: Number of recent embeddings to consider
            mode: Matching mode ('fast' or 'accurate')
                  'fast': Uses only mean embeddings for matching
                  'accurate': Uses mean + recent + top confidence embeddings for better matching
            
        Returns:
            Tuple of (similarity_matrix, best_matches)
            - similarity_matrix: shape (n_embeddings, n_identities) with cosine similarities
            - best_matches: List of (unique_id, confidence) for each embedding
        """
        if not embeddings or not self.identity_clusters:
            return np.array([]), [(None, 0.0) for _ in embeddings]
        
        # Get all identity mean embeddings
        identity_ids = list(self.identity_clusters.keys())
        
        if mode == "accurate":
            # Use comprehensive embedding set for better matching
            identity_embeddings_matrix = []
            for unique_id in identity_ids:
                cluster = self.identity_clusters[unique_id]
                # Combine mean, recent, and top confidence embeddings
                cluster_embeddings = []
                if cluster.mean_embedding is not None:
                    cluster_embeddings.append(cluster.mean_embedding)
                # Add recent embeddings
                recent_embeddings = cluster.all_embeddings[-n_recent_embeddings:]
                cluster_embeddings.extend(recent_embeddings)
                # Add top confidence embeddings
                if cluster.embedding_confidences:
                    top_indices = np.argsort(cluster.embedding_confidences)[-n_top_embed:]
                    for idx in top_indices:
                        if idx < len(cluster.all_embeddings):
                            cluster_embeddings.append(cluster.all_embeddings[idx])
                
                # Calculate average of all selected embeddings
                if cluster_embeddings:
                    avg_embedding = np.mean(cluster_embeddings, axis=0)
                    identity_embeddings_matrix.append(avg_embedding)
                else:
                    identity_embeddings_matrix.append(cluster.mean_embedding)
        else:
            # Fast mode: use only mean embeddings
            identity_embeddings_matrix = [
                self.identity_clusters[unique_id].mean_embedding 
                for unique_id in identity_ids 
                if self.identity_clusters[unique_id].mean_embedding is not None
            ]
        
        if not identity_embeddings_matrix:
            return np.array([]), [(None, 0.0) for _ in embeddings]
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(identity_ids)))
        for i, embedding in enumerate(embeddings):
            for j, identity_embedding in enumerate(identity_embeddings_matrix):
                if identity_embedding is not None:
                    similarity = 1 - cosine(embedding, identity_embedding)
                    similarity_matrix[i, j] = similarity
        
        # Find best matches with exclusive assignment (no identity can be assigned to multiple tracks)
        best_matches = []
        used_identity_indices = set()  # Track which identities have been assigned

        # Get previous IDs of the tracks for stickiness
        previous_ids_per_trackid = {}
        for unique_id, cluster in self.identity_clusters.items():
            for track_id in cluster.associated_track_ids:
                previous_ids_per_trackid[track_id] = unique_id

        # Get track indices sorted by their best similarity score (descending)
        track_best_similarities = []
        for i, embedding_similarities in enumerate(similarity_matrix):
            best_sim = np.max(embedding_similarities)
            track_best_similarities.append((i, best_sim))
        
        # Sort tracks by best similarity (highest first) to prioritize better matches
        track_best_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize all matches as None
        best_matches = [(None, 0.0) for _ in range(len(similarity_matrix))]
        
        # Assign identities in order of best similarity, ensuring exclusive assignment
        for track_idx, _ in track_best_similarities:
            track_id = track_ids[track_idx] if track_ids else track_idx
            
            # Get similarities for this track
            similarities = similarity_matrix[track_idx]
            
            # Find best available identity (not yet assigned)
            best_identity_idx = None
            best_similarity = self.similarity_threshold
            
            # Check if this track had a previous identity and if it's still good enough
            previous_unique_id = previous_ids_per_trackid.get(track_id)
            if previous_unique_id in identity_ids:
                prev_identity_idx = identity_ids.index(previous_unique_id)
                if prev_identity_idx not in used_identity_indices:
                    prev_similarity = similarities[prev_identity_idx]
                    
                    # Check if the previous identity is still within the stickiness margin
                    best_available_sim = np.max([sim for j, sim in enumerate(similarities) if j not in used_identity_indices])
                    if prev_similarity >= best_available_sim - self.track_identity_stickiness_margin:
                        best_identity_idx = prev_identity_idx
                        best_similarity = prev_similarity
            
            # If no sticky assignment, find the best available identity
            if best_identity_idx is None:
                for j, similarity in enumerate(similarities):
                    if j not in used_identity_indices and similarity > best_similarity:
                        best_identity_idx = j
                        best_similarity = similarity
            
            # Assign if a good match was found
            if best_identity_idx is not None:
                unique_id = identity_ids[best_identity_idx]
                best_matches[track_idx] = (unique_id, best_similarity)
                used_identity_indices.add(best_identity_idx)

        return similarity_matrix, best_matches
    
    def _create_new_identity(self, track_id: int, embeddings: List[np.ndarray], confidence: float, timestamp: float) -> str:
        """
        Create a new identity for a track with collected embeddings.
        
        Args:
            track_id: Track ID to create identity for
            embeddings: List of embeddings for this track
            confidence: Confidence score
            timestamp: Current timestamp
            
        Returns:
            unique_id: The newly created identity ID
        """
        unique_id = f"U{self.next_user_number}"
        self.next_user_number += 1
        
        if self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] Creating new identity {unique_id} for track {track_id}")
        
        # Create new identity cluster
        new_cluster = IdentityCluster(
            unique_id=unique_id,
            creation_timestamp=timestamp,
            last_seen_timestamp=timestamp
        )
        
        self.identity_clusters[unique_id] = new_cluster
        
        # Add all embeddings to the new identity
        for embedding in embeddings:
            self._add_embedding_to_cluster(new_cluster, embedding, confidence)
        
        # Set track mapping
        self.track_id_to_unique_id[track_id] = unique_id
        
        self.total_identities_created += 1
        print(f"[IDENTITY] Created new identity {unique_id} for track {track_id}")
        
        return unique_id
    
    def _update_existing_identity(self, unique_id: str, track_id: int, 
                                 embedding: np.ndarray, confidence: float, timestamp: float,
                                 embedding_similarity: float = None):
        """
        Update an existing identity with new embedding data.
        
        Args:
            unique_id: The identity to update
            track_id: Current track ID
            embedding: Face embedding vector
            confidence: Confidence of the embedding
            timestamp: Current timestamp
            embedding_similarity: Similarity score between this embedding and the identity (if available)
        """
        if unique_id not in self.identity_clusters:
            return
        
        cluster = self.identity_clusters[unique_id]
        
        # Check if embedding should be included in the cluster based on similarity
        should_include_embedding = True
        if embedding_similarity is not None:
            should_include_embedding = embedding_similarity >= self.embedding_inclusion_threshold
        elif cluster.mean_embedding is not None:
            similarity = 1 - cosine(embedding, cluster.mean_embedding)
            should_include_embedding = similarity >= self.embedding_inclusion_threshold
        
        # Only add embedding to cluster if it meets the inclusion threshold
        if should_include_embedding:
            self._add_embedding_to_cluster(cluster, embedding, confidence)
        
        # Always update associations, timestamps and statistics regardless of embedding inclusion
        cluster.associated_track_ids.add(track_id)
        cluster.current_track_id = track_id
        cluster.last_seen_timestamp = timestamp
        cluster.total_detections += 1
        
        # Update quality score
        cluster.quality_score = self._calculate_quality_score(cluster)
        
        # Update track_id mapping
        self.track_id_to_unique_id[track_id] = unique_id
    
    def _add_embedding_to_cluster(self, cluster: IdentityCluster, embedding: np.ndarray, confidence: float):
        """Add an embedding to an identity cluster."""
        # Ensure embedding is normalized before storing
        embedding = self._normalize_embedding(embedding)
        
        # Add to cluster
        cluster.all_embeddings.append(embedding)
        cluster.embedding_confidences.append(confidence)
        
        # Maintain maximum embeddings per identity
        if len(cluster.all_embeddings) > self.max_embeddings_per_identity:
            cluster.all_embeddings.pop(0)
            cluster.embedding_confidences.pop(0)
        
        # Update mean embedding
        if self.use_ewma_for_mean and cluster.mean_embedding is not None:
            # EWMA update: new_mean = alpha * new_embedding + (1-alpha) * old_mean
            cluster.mean_embedding = self.ewma_alpha * embedding + (1 - self.ewma_alpha) * cluster.mean_embedding
        else:
            # Traditional averaging
            cluster.mean_embedding = np.mean(cluster.all_embeddings, axis=0)
    
    def _calculate_quality_score(self, cluster: IdentityCluster) -> float:
        """Calculate quality score for an identity cluster."""
        if len(cluster.all_embeddings) < 2:
            return 0.0
        
        # Consistency score based on embedding similarity
        similarities = []
        mean_emb = cluster.mean_embedding
        
        for emb in cluster.all_embeddings[-10:]:  # Use last 10 embeddings
            similarity = 1 - cosine(emb, mean_emb)
            similarities.append(similarity)
        
        consistency_score = np.mean(similarities) if similarities else 0.0
        
        # Detection count score
        detection_score = min(cluster.total_detections / 50.0, 1.0)  # Normalize to 0-1
        
        # Time stability score
        time_active = cluster.last_seen_timestamp - cluster.creation_timestamp
        time_score = min(time_active / 60.0, 1.0)  # Normalize to 0-1 (60 seconds max)
        
        # Combined quality score
        quality = 0.5 * consistency_score + 0.3 * detection_score + 0.2 * time_score
        return quality
    
    def _check_and_perform_merges_batch(self, mode: str = 'fast', n_recent_embeddings: int = 16, n_top_embed: int = 10):
        """Check for potential identity merges and perform them if needed.""" 
        # This is a simplified version - full implementation would check for identities that should be merged
        pass
    
    def cleanup_inactive_identities(self):
        """Remove identities that have been inactive for too long."""
        current_time = time.time()
        inactive_identities = []
        
        for unique_id, cluster in self.identity_clusters.items():
            time_since_last_seen = current_time - cluster.last_seen_timestamp
            if time_since_last_seen > self.identity_timeout:
                inactive_identities.append(unique_id)
        
        for unique_id in inactive_identities:
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] Removing inactive identity {unique_id}")
            del self.identity_clusters[unique_id]
            # Remove from track mappings
            tracks_to_remove = []
            for track_id, mapped_unique_id in self.track_id_to_unique_id.items():
                if mapped_unique_id == unique_id:
                    tracks_to_remove.append(track_id)
            for track_id in tracks_to_remove:
                del self.track_id_to_unique_id[track_id]
    
    def get_unique_id_for_track(self, track_id: int) -> Optional[str]:
        """Get the unique ID assigned to a track."""
        return self.track_id_to_unique_id.get(track_id)
    
    def get_identity_info(self, unique_id: str) -> Optional[Dict]:
        """Get information about a specific identity."""
        if unique_id not in self.identity_clusters:
            return None
        
        cluster = self.identity_clusters[unique_id]
        return {
            "unique_id": unique_id,
            "creation_timestamp": cluster.creation_timestamp,
            "last_seen_timestamp": cluster.last_seen_timestamp,
            "total_detections": cluster.total_detections,
            "quality_score": cluster.quality_score,
            "num_embeddings": len(cluster.all_embeddings),
            "associated_track_ids": list(cluster.associated_track_ids),
            "current_track_id": cluster.current_track_id,
            "custom_name": cluster.custom_name,
            "metadata": cluster.metadata
        }
    
    def get_all_identities(self) -> Dict[str, Dict]:
        """Get information about all identities."""
        return {unique_id: self.get_identity_info(unique_id) for unique_id in self.identity_clusters.keys()}
    
    def get_statistics(self) -> Dict:
        """Get identity management statistics."""
        return {
            "total_identities": len(self.identity_clusters),
            "total_identities_created": self.total_identities_created,
            "total_re_identifications": self.total_re_identifications,
            "total_identity_merges": self.total_identity_merges,
            "active_tracks": len(self.track_id_to_unique_id)
        }
    
    def set_custom_name(self, unique_id: str, custom_name: str) -> bool:
        """Set a custom name for an identity."""
        if unique_id in self.identity_clusters:
            self.identity_clusters[unique_id].custom_name = custom_name
            return True
        return False
    
    def save_identity_database(self):
        """Save identities to persistent storage."""
        if not self.identity_database_path:
            return
        
        # Convert identities to serializable format
        data = {}
        for unique_id, cluster in self.identity_clusters.items():
            data[unique_id] = {
                "creation_timestamp": cluster.creation_timestamp,
                "last_seen_timestamp": cluster.last_seen_timestamp,
                "total_detections": cluster.total_detections,
                "quality_score": cluster.quality_score,
                "custom_name": cluster.custom_name,
                "metadata": cluster.metadata,
                "embeddings": [emb.tolist() for emb in cluster.all_embeddings],
                "embedding_confidences": cluster.embedding_confidences,
                "mean_embedding": cluster.mean_embedding.tolist() if cluster.mean_embedding is not None else None
            }
        
        try:
            os.makedirs(os.path.dirname(self.identity_database_path), exist_ok=True)
            with open(self.identity_database_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[INFO] Saved {len(data)} identities to {self.identity_database_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save identity database: {e}")
    
    def load_identity_database(self):
        """Load identities from persistent storage."""
        if not self.identity_database_path or not os.path.exists(self.identity_database_path):
            return
        
        try:
            with open(self.identity_database_path, 'r') as f:
                data = json.load(f)
            
            for unique_id, cluster_data in data.items():
                cluster = IdentityCluster(
                    unique_id=unique_id,
                    creation_timestamp=cluster_data["creation_timestamp"],
                    last_seen_timestamp=cluster_data["last_seen_timestamp"],
                    total_detections=cluster_data["total_detections"],
                    quality_score=cluster_data["quality_score"],
                    custom_name=cluster_data.get("custom_name"),
                    metadata=cluster_data.get("metadata", {})
                )
                
                # Restore embeddings
                cluster.all_embeddings = [np.array(emb) for emb in cluster_data["embeddings"]]
                cluster.embedding_confidences = cluster_data["embedding_confidences"]
                if cluster_data["mean_embedding"]:
                    cluster.mean_embedding = np.array(cluster_data["mean_embedding"])
                
                self.identity_clusters[unique_id] = cluster
                
                # Update next user number
                if unique_id.startswith('U'):
                    try:
                        user_num = int(unique_id[1:]) + 1
                        self.next_user_number = max(self.next_user_number, user_num)
                    except ValueError:
                        pass
            
            print(f"[INFO] Loaded {len(data)} identities from {self.identity_database_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load identity database: {e}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize an embedding vector."""
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
