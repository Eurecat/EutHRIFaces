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
                                       mode: str = 'fast', n_recent_embeddings: int = 20, n_top_embed: int = 10) -> Tuple[np.ndarray, List[Tuple[Optional[str], float]]]:
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
        """
        Check all identities for potential merges using batch processing similar to _find_best_identity_match_batch.
        This compares all identities against each other using similarity matrices.
        
        Args:
            mode: Matching mode ('fast' or 'accurate')
                  'fast': Uses only mean embeddings for comparison
                  'accurate': Uses mean + recent + top confidence embeddings for better comparison
            n_recent_embeddings: Number of recent embeddings to consider in accurate mode
            n_top_embed: Number of top confidence embeddings to consider in accurate mode
        """
        if len(self.identity_clusters) < 2:
            return  # Need at least 2 identities to merge
        
        # Get all identity IDs and their clusters
        identity_ids = list(self.identity_clusters.keys())
        
        # Filter out identities without embeddings
        valid_identity_ids = []
        valid_clusters = []
        for unique_id in identity_ids:
            cluster = self.identity_clusters[unique_id]
            if cluster.mean_embedding is not None and len(cluster.all_embeddings) > 0:
                valid_identity_ids.append(unique_id)
                valid_clusters.append(cluster)
        
        if len(valid_clusters) < 2:
            return  # Need at least 2 valid identities to merge
        
        if self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] Batch merge check: comparing {len(valid_clusters)} identities")
            print(f"[IDENTITY_DEBUG] Identity IDs: {valid_identity_ids}")
        
        # Use clustering threshold for merges
        merge_threshold = max(0.0, self.clustering_threshold)
        if self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] Using merge threshold: {merge_threshold:.3f}")
        
        # Compute similarity matrix between all identities
        if mode == "accurate":
            # Accurate mode: compute 3 types of similarities (mean, recent, top confidence)
            mean_embeddings = []
            recent_embeddings_lists = []  # List of lists for recent embeddings
            top_conf_embeddings_lists = []  # List of lists for top confidence embeddings
            
            for cluster in valid_clusters:
                # 1. Mean embedding
                mean_embeddings.append(cluster.mean_embedding)
                
                # 2. Recent embeddings (last n_recent_embeddings)
                recent_embeddings = cluster.all_embeddings[-n_recent_embeddings:]
                recent_embeddings_lists.append(recent_embeddings)
                
                # 3. Top confidence embeddings (top n_top_embed)
                if len(cluster.embedding_confidences) > 0:
                    conf_emb_pairs = list(zip(cluster.embedding_confidences, cluster.all_embeddings))
                    conf_emb_pairs.sort(key=lambda x: x[0], reverse=True)  # Sort by confidence
                    top_embeddings = [emb for _, emb in conf_emb_pairs[:n_top_embed]]
                else:
                    top_embeddings = cluster.all_embeddings[:n_top_embed] if len(cluster.all_embeddings) >= n_top_embed else cluster.all_embeddings
                top_conf_embeddings_lists.append(top_embeddings)
            
            # Convert to numpy arrays for efficient computation
            mean_identities_matrix = np.array(mean_embeddings)  # shape: (n_identities, embedding_dim)
            
            # Compute 3 similarity matrices
            # 1. Mean similarity matrix (identity x identity)
            mean_similarity_matrix = np.dot(mean_identities_matrix, mean_identities_matrix.T)
            
            # 2. Recent embeddings similarity matrix (max similarity with recent embeddings)
            recent_similarity_matrix = np.zeros((len(valid_clusters), len(valid_clusters)))
            for i, recent_embs_i in enumerate(recent_embeddings_lists):
                for j, recent_embs_j in enumerate(recent_embeddings_lists):
                    if i != j and recent_embs_i and recent_embs_j:
                        recent_matrix_i = np.array(recent_embs_i)  # shape: (n_recent_i, embedding_dim)
                        recent_matrix_j = np.array(recent_embs_j)  # shape: (n_recent_j, embedding_dim)
                        # Compute similarity between all pairs and take max
                        similarity_cross = np.dot(recent_matrix_i, recent_matrix_j.T)  # shape: (n_recent_i, n_recent_j)
                        recent_similarity_matrix[i, j] = np.max(similarity_cross)
            
            # 3. Top confidence embeddings similarity matrix
            top_conf_similarity_matrix = np.zeros((len(valid_clusters), len(valid_clusters)))
            for i, top_conf_embs_i in enumerate(top_conf_embeddings_lists):
                for j, top_conf_embs_j in enumerate(top_conf_embeddings_lists):
                    if i != j and top_conf_embs_i and top_conf_embs_j:
                        top_conf_matrix_i = np.array(top_conf_embs_i)  # shape: (n_top_i, embedding_dim)
                        top_conf_matrix_j = np.array(top_conf_embs_j)  # shape: (n_top_j, embedding_dim)
                        # Compute similarity between all pairs and take max
                        similarity_cross = np.dot(top_conf_matrix_i, top_conf_matrix_j.T)  # shape: (n_top_i, n_top_j)
                        top_conf_similarity_matrix[i, j] = np.max(similarity_cross)
            
            # Combined similarity matrix: weighted combination of the 3 matrices
            combined_similarity_matrix = (0.5 * mean_similarity_matrix + 
                                        0.3 * recent_similarity_matrix + 
                                        0.2 * top_conf_similarity_matrix)
            
            similarity_matrix = combined_similarity_matrix
            
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] Accurate mode: computed combined similarity matrix shape: {similarity_matrix.shape}")
        
        else:
            # Fast mode: Use only mean embeddings
            mean_embeddings = [cluster.mean_embedding for cluster in valid_clusters]
            
            # Convert to numpy arrays for efficient computation
            identities_matrix = np.array(mean_embeddings)  # shape: (n_identities, embedding_dim)
            
            # Compute cosine similarity matrix: identities x identities
            similarity_matrix = np.dot(identities_matrix, identities_matrix.T)
            
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] Fast mode: computed similarity matrix shape: {similarity_matrix.shape}")
        
        # Find merge candidates
        merge_candidates = []
        
        # Iterate through upper triangle of similarity matrix (avoid duplicates and self-comparisons)
        for i in range(len(valid_clusters)):
            for j in range(i + 1, len(valid_clusters)):
                similarity = similarity_matrix[i, j]
                
                if self.enable_debug_prints:
                    if similarity < 0.2:
                        continue  # Skip printing for very low similarity
                    elif 0.2 <= similarity < 0.3:
                        print(f"\033[93m[IDENTITY_DEBUG] Comparing {valid_identity_ids[i]} <-> {valid_identity_ids[j]}: similarity={similarity:.3f}\033[0m")  # Yellow
                    elif 0.3 <= similarity < 0.4:
                        print(f"\033[33m[IDENTITY_DEBUG] Comparing {valid_identity_ids[i]} <-> {valid_identity_ids[j]}: similarity={similarity:.3f}\033[0m")  # Orange
                    else:
                        print(f"\033[92m[IDENTITY_DEBUG] Comparing {valid_identity_ids[i]} <-> {valid_identity_ids[j]}: similarity={similarity:.3f}\033[0m")  # Green
                
                if similarity > merge_threshold:
                    merge_candidates.append((valid_identity_ids[i], valid_identity_ids[j], similarity))
                    if self.enable_debug_prints:
                        print(f"[IDENTITY_DEBUG] Merge candidate: {valid_identity_ids[i]} <-> {valid_identity_ids[j]} (similarity: {similarity:.3f})")
        
        # Perform merges with best candidates first
        if merge_candidates:
            # Sort by similarity (best first)
            merge_candidates.sort(key=lambda x: x[2], reverse=True)
            
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] Found {len(merge_candidates)} merge candidates")
            
            # Keep track of already merged identities to avoid conflicts
            merged_identities = set()
            
            for id1, id2, similarity in merge_candidates:
                # Skip if either identity was already merged
                if id1 in merged_identities or id2 in merged_identities:
                    continue
                
                # Check if both identities still exist (might have been merged already)
                if id1 not in self.identity_clusters or id2 not in self.identity_clusters:
                    continue
                
                # Always merge into the older identity (lower USER number or older timestamp)
                id1_num = self._extract_user_number(id1)
                id2_num = self._extract_user_number(id2)
                
                if id1_num is not None and id2_num is not None:
                    if id1_num < id2_num:
                        # Keep id1, merge id2 into it
                        success = self.merge_identities(id1, id2)
                        if success:
                            merged_identities.add(id2)
                            if self.enable_debug_prints:
                                print(f"[IDENTITY] Batch auto-merged {id2} into {id1} (similarity: {similarity:.3f})")
                    else:
                        # Keep id2, merge id1 into it
                        success = self.merge_identities(id2, id1)
                        if success:
                            merged_identities.add(id1)
                            if self.enable_debug_prints:
                                print(f"[IDENTITY] Batch auto-merged {id1} into {id2} (similarity: {similarity:.3f})")
                else:
                    # Fallback: merge into older identity by timestamp
                    cluster1 = self.identity_clusters[id1]
                    cluster2 = self.identity_clusters[id2]
                    
                    if cluster1.creation_timestamp < cluster2.creation_timestamp:
                        success = self.merge_identities(id1, id2)
                        if success:
                            merged_identities.add(id2)
                            if self.enable_debug_prints:
                                print(f"[IDENTITY] Batch auto-merged {id2} into {id1} (by timestamp)")
                    else:
                        success = self.merge_identities(id2, id1)
                        if success:
                            merged_identities.add(id1)
                            if self.enable_debug_prints:
                                print(f"[IDENTITY] Batch auto-merged {id1} into {id2} (by timestamp)")
            
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] Batch merge complete. Merged {len(merged_identities)} identities: {list(merged_identities)}")
        elif self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] No merge candidates found above threshold {merge_threshold:.3f}")

    def merge_identities(self, primary_id: str, secondary_id: str) -> bool:
        """
        Merge two identities into one.
        
        Args:
            primary_id: The identity to keep
            secondary_id: The identity to merge into primary
            
        Returns:
            True if merge was successful
        """
        if primary_id not in self.identity_clusters or secondary_id not in self.identity_clusters:
            return False
        
        primary_cluster = self.identity_clusters[primary_id]
        secondary_cluster = self.identity_clusters[secondary_id]
        
        # Merge embeddings - ensure all are normalized
        # Normalize secondary embeddings before merging
        normalized_secondary_embeddings = []
        for emb in secondary_cluster.all_embeddings:
            normalized_emb = self._normalize_embedding(emb)
            normalized_secondary_embeddings.append(normalized_emb)
        
        primary_cluster.all_embeddings.extend(normalized_secondary_embeddings)
        primary_cluster.embedding_confidences.extend(secondary_cluster.embedding_confidences)
        
        # Limit total embeddings
        if len(primary_cluster.all_embeddings) > self.max_embeddings_per_identity:
            # Keep most recent embeddings
            primary_cluster.all_embeddings = primary_cluster.all_embeddings[-self.max_embeddings_per_identity:]
            primary_cluster.embedding_confidences = primary_cluster.embedding_confidences[-self.max_embeddings_per_identity:]
        
        # Recalculate mean embedding using traditional averaging (not EWMA)
        # When merging clusters, we want to compute the true mean of all embeddings
        primary_cluster.mean_embedding = np.mean(primary_cluster.all_embeddings, axis=0)
        primary_cluster.mean_embedding = self._normalize_embedding(primary_cluster.mean_embedding)
        
        # Merge track ID associations
        primary_cluster.associated_track_ids.update(secondary_cluster.associated_track_ids)
        primary_cluster.current_track_id = secondary_cluster.current_track_id

        # Update statistics
        primary_cluster.total_detections += secondary_cluster.total_detections
        primary_cluster.creation_timestamp = min(primary_cluster.creation_timestamp, secondary_cluster.creation_timestamp)
        
        # Update track_id mappings
        for track_id in secondary_cluster.associated_track_ids:
            if track_id in self.track_id_to_unique_id:
                self.track_id_to_unique_id[track_id] = primary_id
        
        # Remove secondary identity
        del self.identity_clusters[secondary_id]
        
        # Update quality score
        primary_cluster.quality_score = self._calculate_quality_score(primary_cluster)
        
        self.total_identity_merges += 1
        print(f"[IDENTITY] Merged {secondary_id} into {primary_id}")
        
        return True

    def cleanup_inactive_track_mappings(self, current_active_track_ids: Set[int]):
        """
        Clean up track_id mappings for tracks that are no longer active.
        
        This is critical to prevent track_id reuse issues where a new track
        gets the same track_id as a previous track and inherits the wrong identity.
        
        Args:
            current_active_track_ids: Set of track IDs that are currently active in the tracker
        """
        inactive_track_ids = []
        
        # Find track IDs that are mapped but no longer active
        for track_id in self.track_id_to_unique_id.keys():
            if track_id not in current_active_track_ids:
                inactive_track_ids.append(track_id)
        
        if self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] Inactive track IDs to clean: {inactive_track_ids}")
        
        # Remove mappings for inactive tracks
        for track_id in inactive_track_ids:
            unique_id = self.track_id_to_unique_id[track_id]
            del self.track_id_to_unique_id[track_id]
            
            # Remove from cluster's current track ID (but keep in associated for history)
            if unique_id in self.identity_clusters:
                self.identity_clusters[unique_id].current_track_id = -1
            
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] Cleaned up mapping: track {track_id} -> {unique_id} (track no longer active)")
        
        # Clean up pending identities for inactive tracks
        pending_to_remove = []
        for track_id in self.pending_identities.keys():
            if track_id not in current_active_track_ids:
                pending_to_remove.append(track_id)
        
        for track_id in pending_to_remove:
            del self.pending_identities[track_id]
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] Cleaned up pending identity for inactive track {track_id}")
        
        if inactive_track_ids and self.enable_debug_prints:
            print(f"[IDENTITY_DEBUG] Cleaned up {len(inactive_track_ids)} inactive track mappings")
            print(f"[IDENTITY_DEBUG] Active tracks: {sorted(current_active_track_ids)}")
            print(f"[IDENTITY_DEBUG] Remaining mappings: {dict(sorted(self.track_id_to_unique_id.items(), key=lambda x: str(x[0])))}")
            print(f"[IDENTITY_DEBUG] Current identity clusters: {list(self.identity_clusters.keys())}")

    def _find_track_for_identity(self, unique_id: str) -> Optional[int]:
        """
        Find the track ID that is currently assigned to a given identity.
        
        Args:
            unique_id: The identity to search for
            
        Returns:
            track_id if found, None otherwise
        """
        found_tracks = []
        for track_id, assigned_id in self.track_id_to_unique_id.items():
            if assigned_id == unique_id:
                found_tracks.append(track_id)
        
        if len(found_tracks) > 1:
            # Multiple tracks found for same identity - this is a violation!
            if self.enable_debug_prints:
                print(f"[IDENTITY_DEBUG] WARNING: _find_track_for_identity found multiple tracks for {unique_id}: {found_tracks}")
            # Return the first one, but this indicates a problem
            return found_tracks[0]
        elif len(found_tracks) == 1:
            return found_tracks[0]
        else:
            return None

    def _get_fixed_color_for_identity(self, unique_id: str) -> str:
        """
        Get a fixed color for the given identity. If the identity doesn't have a color assigned yet,
        assign one from the available colors pool.
        
        Args:
            unique_id: The unique identity ID
            
        Returns:
            str: Color name for this identity
        """
        if unique_id not in self.identity_color_mapping:
            # Extract user number for consistent ordering if possible
            user_number = self._extract_user_number(unique_id)
            if user_number is not None:
                # Use user number to pick color consistently
                color_index = (user_number - 1) % len(self.available_colors)
                self.identity_color_mapping[unique_id] = self.available_colors[color_index]
            else:
                # For non-standard unique_ids, use the next available color
                used_colors = set(self.identity_color_mapping.values())
                available = [c for c in self.available_colors if c not in used_colors]
                if available:
                    self.identity_color_mapping[unique_id] = available[0]
                else:
                    # Fallback to cycling through colors if all are used
                    color_index = len(self.identity_color_mapping) % len(self.available_colors)
                    self.identity_color_mapping[unique_id] = self.available_colors[color_index]
        
        return self.identity_color_mapping[unique_id]

    def plot_identity_clusters_embeddings(self):
        """
        Plot the identity clusters embeddings using dimensionality reduction.
        Creates a 2D visualization of all embeddings colored by identity with quality-based styling.
        """
        if not _SKLEARN_AVAILABLE:
            print("[PLOT] Plotting requires sklearn and matplotlib. Install with: pip install scikit-learn matplotlib")
            return
            
        try:
            if not self.identity_clusters:
                print("[PLOT] No identity clusters to plot")
                return
            
            # Collect all embeddings with quality information
            all_embeddings = []
            identity_labels = []
            embedding_qualities = []  # True if above inclusion threshold, False otherwise
            embedding_types = []  # 'regular' or 'mean'
            
            for unique_id, cluster in self.identity_clusters.items():
                if cluster.all_embeddings and cluster.mean_embedding is not None:
                    # Add regular embeddings with quality assessment
                    for embedding in cluster.all_embeddings:
                        all_embeddings.append(embedding)
                        identity_labels.append(unique_id)
                        embedding_types.append('regular')
                        
                        # Calculate similarity to mean embedding to determine quality
                        similarity = 1 - cosine(embedding, cluster.mean_embedding)
                        is_high_quality = similarity >= self.embedding_inclusion_threshold
                        embedding_qualities.append(is_high_quality)
                    
                    # Add mean embedding as a separate point
                    all_embeddings.append(cluster.mean_embedding)
                    identity_labels.append(unique_id)
                    embedding_types.append('mean')
                    embedding_qualities.append(True)  # Mean is always considered high quality
            
            if len(all_embeddings) < 2:
                print(f"[PLOT] Not enough embeddings to plot ({len(all_embeddings)})")
                return
            
            # Convert to numpy array
            embeddings_matrix = np.array(all_embeddings)
            print(f"[PLOT] Plotting {len(all_embeddings)} embeddings from {len(self.identity_clusters)} identities")
            
            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings_matrix)
            
            # Create the plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # Plot each identity with fixed colors and quality-based styling
            unique_identities = list(set(identity_labels))
            
            for unique_id in unique_identities:
                # Get fixed color for this identity
                base_color = self._get_fixed_color_for_identity(unique_id)
                
                # Get indices for this identity
                identity_indices = [i for i, label in enumerate(identity_labels) if label == unique_id]
                
                # Separate indices by type and quality
                regular_high_quality_indices = [i for i in identity_indices if embedding_types[i] == 'regular' and embedding_qualities[i]]
                regular_low_quality_indices = [i for i in identity_indices if embedding_types[i] == 'regular' and not embedding_qualities[i]]
                mean_indices = [i for i in identity_indices if embedding_types[i] == 'mean']
                
                # Plot high-quality regular embeddings (normal color, full opacity)
                if regular_high_quality_indices:
                    x_coords = [embeddings_2d[i, 0] for i in regular_high_quality_indices]
                    y_coords = [embeddings_2d[i, 1] for i in regular_high_quality_indices]
                    plt.scatter(x_coords, y_coords, c=base_color, alpha=0.8, s=50, 
                               label=f"{unique_id} high-quality ({len(regular_high_quality_indices)})")
                
                # Plot low-quality regular embeddings (lighter color, reduced opacity)
                if regular_low_quality_indices:
                    x_coords = [embeddings_2d[i, 0] for i in regular_low_quality_indices]
                    y_coords = [embeddings_2d[i, 1] for i in regular_low_quality_indices]
                    plt.scatter(x_coords, y_coords, c=base_color, alpha=0.3, s=40,
                               label=f"{unique_id} low-quality ({len(regular_low_quality_indices)})")
                
                # Plot mean embeddings (darker color, larger size)
                if mean_indices:
                    x_coords = [embeddings_2d[i, 0] for i in mean_indices]
                    y_coords = [embeddings_2d[i, 1] for i in mean_indices]
                    plt.scatter(x_coords, y_coords, c=base_color, alpha=1.0, s=120, marker='*',
                               label=f"{unique_id} mean", edgecolors='black', linewidth=1)
            
            plt.title(f"Identity Clusters Embeddings Visualization (PCA)\nInclusion threshold: {self.embedding_inclusion_threshold:.2f}")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"/tmp/face_recognition_clusters_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
            print(f"[PLOT] Identity clusters plot saved to: {plot_filename}")
            plt.close()
            
        except Exception as e:
            print(f"[PLOT] Error creating identity clusters plot: {e}")

    def _extract_user_number(self, unique_id: str) -> Optional[int]:
        """Extract the user number from a unique ID like 'U1', 'U2', etc."""
        if unique_id.startswith('U') and unique_id[1:].isdigit():
            return int(unique_id[1:])
        return None

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
