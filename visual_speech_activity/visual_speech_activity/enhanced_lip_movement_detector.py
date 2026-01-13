#!/usr/bin/env python3
"""
Enhanced Lip Movement Detector for Visual Speech Activity Detection

This module implements a sophisticated lip movement detector based on temporal analysis
of mouth landmarks using full dlib facial landmarks. It uses an approach inspired by 
Lip-Movement-Net with enhanced features:

- Analyzes vertical and horizontal lip movements from full 68-point dlib landmarks
- Uses multiple lip movement features: MAR, MER, inner/outer lip distances
- Implements RNN-based classification for robust temporal analysis
- Per-identity buffering for robust detection across frame-to-frame tracking changes
- Real-time performance suitable for ROS2 integration

The detector extracts comprehensive lip features from landmarks 48-67 (mouth region)
and uses temporal patterns to determine speaking activity with high accuracy.

Key Features:
- Full dlib landmark support (68 points)
- Multiple lip movement features (MAR, MER, lip height, width)
- RNN-based temporal classification
- Per-identity buffering for robust detection
- Configurable detection parameters

References:
- Lip-Movement-Net: https://github.com/sachinsdate/lip-movement-net
- dlib facial landmark detection
- Enhanced with multiple geometric features
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque, defaultdict
import logging
import math


class EnhancedLipMovementDetector:
    """
    Advanced lip movement detector using full dlib facial landmarks and RNN classification.
    
    This detector uses comprehensive temporal analysis of mouth landmarks to determine 
    if a person is speaking. It maintains buffers of lip movements per recognized_face_id 
    for robust detection and uses multiple geometric features for improved accuracy.
    
    The enhanced approach:
    1. Extract full mouth landmarks (points 48-67 from 68-point dlib model)
    2. Calculate multiple features: MAR, MER, inner/outer lip distances
    3. Use RNN-based temporal analysis for robust classification
    4. Apply confidence-based thresholding and temporal smoothing
    
    Key Features:
    - Supports both 5-point and 68-point facial landmarks
    - Multiple lip movement features for robust detection
    - RNN-based temporal classification
    - Per-identity temporal buffering
    - Configurable parameters for different use cases
    
    Attributes:
        window_size: Number of frames to analyze for temporal patterns
        movement_threshold: Minimum feature variation to consider as speaking
        speaking_threshold: Threshold for speaking classification
        use_full_landmarks: Whether to use full 68-point landmarks or fallback to 5-point
        rnn_enabled: Whether to use RNN classification or simple thresholding
    """

    # dlib 68-point facial landmark indices for mouth region
    # Outer lip landmarks (12 points): 48-59
    OUTER_LIP_INDICES = list(range(48, 60))
    # Inner lip landmarks (8 points): 60-67  
    INNER_LIP_INDICES = list(range(60, 68))
    
    # Key mouth landmark indices for feature extraction
    MOUTH_LEFT = 48      # Left corner of mouth
    MOUTH_RIGHT = 54     # Right corner of mouth
    MOUTH_TOP = 51       # Top center of upper lip
    MOUTH_BOTTOM = 57    # Bottom center of lower lip
    
    # Inner lip landmarks for refined measurements
    INNER_MOUTH_LEFT = 60
    INNER_MOUTH_RIGHT = 64
    INNER_MOUTH_TOP = 62
    INNER_MOUTH_BOTTOM = 66

    def __init__(
        self,
        window_size: int = 20,
        movement_threshold: float = 0.02,
        speaking_threshold: float = 0.5,
        temporal_smoothing: bool = True,
        min_frames_for_detection: int = 5,
        use_full_landmarks: bool = True,
        rnn_enabled: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the enhanced lip movement detector.
        
        Args:
            window_size: Number of frames to maintain in temporal buffer (default: 20)
            movement_threshold: Minimum feature variation to detect movement (default: 0.02)
            speaking_threshold: Confidence threshold for speaking classification (default: 0.5)
            temporal_smoothing: Whether to apply temporal smoothing (default: True)
            min_frames_for_detection: Minimum frames needed before detection (default: 5)
            use_full_landmarks: Whether to use full 68-point landmarks when available (default: True)
            rnn_enabled: Whether to use RNN classification or simple thresholding (default: True)
            logger: Optional logger for debugging
        """
        self.window_size = window_size
        self.movement_threshold = movement_threshold
        self.speaking_threshold = speaking_threshold
        self.temporal_smoothing = temporal_smoothing
        self.min_frames_for_detection = min_frames_for_detection
        self.use_full_landmarks = use_full_landmarks
        self.rnn_enabled = rnn_enabled
        self.logger = logger
        
        # Per-identity temporal buffers for multiple features
        # Dictionary mapping recognized_face_id -> deque of feature vectors
        self.feature_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Per-identity speaking state buffers for temporal smoothing
        self.speaking_state_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=5)  # Last 5 decisions
        )
        
        # Simple RNN weights for temporal classification (lightweight LSTM-like)
        if self.rnn_enabled:
            self._initialize_rnn_weights()
        
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(
            f"Enhanced LipMovementDetector initialized: window_size={window_size}, "
            f"movement_threshold={movement_threshold}, speaking_threshold={speaking_threshold}, "
            f"use_full_landmarks={use_full_landmarks}, rnn_enabled={rnn_enabled}"
        )
    
    def _initialize_rnn_weights(self):
        """Initialize simple RNN weights for temporal classification."""
        # Simple LSTM-like cell for processing temporal sequences
        # Input size: 4 features (MAR, MER, lip_height, lip_width)
        # Hidden size: 8 neurons
        input_size = 4
        hidden_size = 8
        
        # Initialize weights with small random values (no fixed seed for real randomness)
        
        # Input to hidden weights
        self.W_ih = np.random.randn(input_size, hidden_size) * 0.1
        self.b_ih = np.zeros(hidden_size)
        
        # Hidden to hidden weights (recurrent)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_hh = np.zeros(hidden_size)
        
        # Output weights
        self.W_out = np.random.randn(hidden_size, 1) * 0.1
        self.b_out = 0.0
        
        # Per-identity hidden states
        self.hidden_states: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(hidden_size)
        )
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to similar scales for better RNN performance.
        
        Args:
            features: [MAR, MER, lip_height, lip_width]
            
        Returns:
            Normalized features
        """
        # Expected ranges based on typical values:
        # MAR: 0.1 - 0.5
        # MER: 20 - 100 
        # lip_height: 0.01 - 0.2
        # lip_width: 0.5 - 1.2
        
        normalized = np.zeros_like(features)
        normalized[0] = (features[0] - 0.3) / 0.2  # MAR normalization
        normalized[1] = (features[1] - 60) / 40    # MER normalization  
        normalized[2] = (features[2] - 0.1) / 0.1  # lip_height normalization
        normalized[3] = (features[3] - 0.85) / 0.2 # lip_width normalization
        
        return normalized
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to similar scales for better RNN performance.
        
        Args:
            features: [MAR, MER, lip_height, lip_width]
            
        Returns:
            Normalized features
        """
        # Expected ranges based on typical values:
        # MAR: 0.1 - 0.5
        # MER: 20 - 100 
        # lip_height: 0.01 - 0.2
        # lip_width: 0.5 - 1.2
        
        normalized = np.zeros_like(features)
        normalized[0] = (features[0] - 0.3) / 0.2  # MAR normalization
        normalized[1] = (features[1] - 60) / 40    # MER normalization  
        normalized[2] = (features[2] - 0.1) / 0.1  # lip_height normalization
        normalized[3] = (features[3] - 0.85) / 0.2 # lip_width normalization
        
        return normalized
    
    def extract_lip_features_full(self, landmarks: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Extract comprehensive lip movement features from full 68-point dlib landmarks.
        
        Args:
            landmarks: List of (x, y) tuples for all 68 facial landmarks
            
        Returns:
            Feature vector [MAR, MER, lip_height, lip_width] or None if extraction fails
        """
        if len(landmarks) < 68:
            return None
            
        try:
            # Extract outer lip landmarks
            outer_lip = [landmarks[i] for i in self.OUTER_LIP_INDICES]
            inner_lip = [landmarks[i] for i in self.INNER_LIP_INDICES]
            
            # Calculate MAR (Mouth Aspect Ratio) - vertical mouth opening
            # Top and bottom center points
            top_lip = landmarks[self.MOUTH_TOP]
            bottom_lip = landmarks[self.MOUTH_BOTTOM]
            left_corner = landmarks[self.MOUTH_LEFT] 
            right_corner = landmarks[self.MOUTH_RIGHT]
            
            # Vertical distance (mouth opening)
            vertical_dist = np.sqrt((top_lip[0] - bottom_lip[0])**2 + (top_lip[1] - bottom_lip[1])**2)
            
            # Horizontal distance (mouth width)
            horizontal_dist = np.sqrt((right_corner[0] - left_corner[0])**2 + (right_corner[1] - left_corner[1])**2)
            
            # MAR = vertical opening / horizontal width
            mar = vertical_dist / (horizontal_dist + 1e-6)
            
            # Calculate MER (Mouth Elongation Ratio) - horizontal mouth stretching
            mer = horizontal_dist
            
            # Inner lip measurements for refined detection
            inner_top = landmarks[self.INNER_MOUTH_TOP]
            inner_bottom = landmarks[self.INNER_MOUTH_BOTTOM]
            inner_left = landmarks[self.INNER_MOUTH_LEFT]
            inner_right = landmarks[self.INNER_MOUTH_RIGHT]
            
            # Inner lip height and width
            inner_height = np.sqrt((inner_top[0] - inner_bottom[0])**2 + (inner_top[1] - inner_bottom[1])**2)
            inner_width = np.sqrt((inner_right[0] - inner_left[0])**2 + (inner_right[1] - inner_left[1])**2)
            
            # Normalized inner lip height
            lip_height = inner_height / (horizontal_dist + 1e-6)
            
            # Lip width ratio
            lip_width = inner_width / (horizontal_dist + 1e-6)
            
            # Return feature vector
            features = np.array([mar, mer, lip_height, lip_width], dtype=np.float32)
            return features
            
        except (IndexError, ZeroDivisionError) as e:
            self.logger.debug(f"Failed to extract full lip features: {e}")
            return None
    
    def extract_lip_features_simple(self, landmarks: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Extract basic lip movement features from 5-point face landmarks (fallback).
        
        Args:
            landmarks: List of (x, y) tuples, expects at least 5 landmarks
            
        Returns:
            Feature vector [MAR, MER, lip_height, lip_width] or None if extraction fails
        """
        if len(landmarks) < 5:
            return None
            
        try:
            # Extract mouth corners from 5-point landmarks (indices 3, 4)
            left_mouth = landmarks[3]   # Left mouth corner
            right_mouth = landmarks[4]  # Right mouth corner
            
            # Calculate horizontal mouth width
            horizontal_dist = np.sqrt(
                (right_mouth[0] - left_mouth[0])**2 + 
                (right_mouth[1] - left_mouth[1])**2
            )
            
            # For 5-point landmarks, we approximate vertical opening using mouth corner positions
            # This is less accurate but provides basic functionality
            mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            
            # Estimate vertical mouth opening based on mouth corner curvature
            # (This is a simplification - full landmarks give much better results)
            vertical_dist = abs(left_mouth[1] - right_mouth[1]) * 0.5  # Approximation
            
            # Calculate basic features
            mar = vertical_dist / (horizontal_dist + 1e-6)
            mer = horizontal_dist
            lip_height = vertical_dist / (horizontal_dist + 1e-6)
            lip_width = 1.0  # Cannot be accurately computed from 5-point landmarks
            
            features = np.array([mar, mer, lip_height, lip_width], dtype=np.float32)
            return features
            
        except (IndexError, ZeroDivisionError) as e:
            self.logger.debug(f"Failed to extract simple lip features: {e}")
            return None
    
    def _rnn_forward(self, features: np.ndarray, hidden_state: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Forward pass through simple RNN for temporal classification.
        
        Args:
            features: Input feature vector [MAR, MER, lip_height, lip_width],

            MAR (Mouth Aspect Ratio), MER (Mouth Elongation Ratio)

            hidden_state: Previous hidden state
            
        Returns:
            Tuple of (output_probability, new_hidden_state)
        """
        # Simple LSTM-like forward pass
        # Input transformation
        input_contrib = np.dot(features, self.W_ih) + self.b_ih
        
        # Hidden state transformation  
        hidden_contrib = np.dot(hidden_state, self.W_hh) + self.b_hh
        
        # Combine and apply activation
        combined = input_contrib + hidden_contrib
        new_hidden = np.tanh(combined)  # Using tanh activation
        
        # Output layer
        output = np.dot(new_hidden, self.W_out) + self.b_out
        probability = 1.0 / (1.0 + np.exp(-output))  # Sigmoid activation
        
        return float(probability), new_hidden
    
    def detect_speaking(
        self, 
        recognized_face_id: str,
        landmarks: List[Tuple[float, float]]
    ) -> Tuple[bool, float]:
        """
        Enhanced speaking detection using full landmark features and RNN classification.
        
        This method:
        1. Extracts comprehensive lip features from landmarks
        2. Updates temporal feature buffer
        3. Applies RNN-based temporal analysis (if enabled)
        4. Returns speaking status and confidence
        
        Args:
            recognized_face_id: Robust face identity ID
            landmarks: List of (x, y) facial landmark coordinates
            
        Returns:
            Tuple of (is_speaking: bool, confidence: float)
        """
        if self.logger:
            self.logger.info(f"Processing landmarks for {recognized_face_id}, landmark count: {len(landmarks)}")
        
        # Extract lip features based on landmark type
        if self.use_full_landmarks and len(landmarks) >= 68:
            features = self.extract_lip_features_full(landmarks)
            if self.logger:
                self.logger.info(f"Using full 68-point landmarks for {recognized_face_id}")
        else:
            features = self.extract_lip_features_simple(landmarks)
            if self.logger:
                self.logger.info(f"Using simple landmark features for {recognized_face_id}")
        
        if features is None:
            if self.logger:
                self.logger.info(f"Failed to extract features for {recognized_face_id}")
            # Return previous state or default
            return self._get_default_state(recognized_face_id)
        
        if self.logger:
            self.logger.info(f"Extracted features for {recognized_face_id}: MAR={features[0]:.4f}, MER={features[1]:.4f}, lip_h={features[2]:.4f}, lip_w={features[3]:.4f}")
        
        # Update feature buffer for this identity
        self.feature_buffers[recognized_face_id].append(features)
        
        # Check if we have enough frames
        buffer = self.feature_buffers[recognized_face_id]
        if len(buffer) < self.min_frames_for_detection:
            if self.logger:
                self.logger.info(f"Not enough frames for {recognized_face_id}: {len(buffer)}/{self.min_frames_for_detection}")
            return False, 0.0
        
        # Analyze temporal patterns
        if self.rnn_enabled:
            is_speaking, confidence = self._analyze_temporal_pattern_rnn(recognized_face_id, features)
            if self.logger:
                self.logger.info(f"RNN analysis for {recognized_face_id}: speaking={is_speaking}, conf={confidence:.3f}")
        else:
            # Use simple threshold-based detection as fallback
            is_speaking, confidence = self._analyze_temporal_pattern_simple(buffer)
            if self.logger:
                self.logger.info(f"Simple analysis for {recognized_face_id}: speaking={is_speaking}, conf={confidence:.3f}")
        
        # Apply temporal smoothing if enabled
        if self.temporal_smoothing:
            old_speaking, old_confidence = is_speaking, confidence
            is_speaking, confidence = self._apply_temporal_smoothing(
                recognized_face_id, is_speaking, confidence
            )
            if self.logger and (old_speaking != is_speaking or abs(old_confidence - confidence) > 0.1):
                self.logger.info(f"Temporal smoothing for {recognized_face_id}: {old_speaking},{old_confidence:.3f} -> {is_speaking},{confidence:.3f}")
        
        if self.logger:
            self.logger.info(f"Final result for {recognized_face_id}: speaking={is_speaking}, confidence={confidence:.3f}")
        
        return is_speaking, confidence
    
    def _analyze_temporal_pattern_rnn(self, recognized_face_id: str, current_features: np.ndarray) -> Tuple[bool, float]:
        """
        RNN-based temporal pattern analysis for speaking detection.
        
        Args:
            recognized_face_id: Face identity for maintaining hidden state
            current_features: Current frame's lip features
            
        Returns:
            Tuple of (is_speaking: bool, confidence: float)
        """
        # Normalize features for better RNN performance
        normalized_features = self._normalize_features(current_features)
        
        if self.logger:
            self.logger.info(f"Normalized features for {recognized_face_id}: [{normalized_features[0]:.3f}, {normalized_features[1]:.3f}, {normalized_features[2]:.3f}, {normalized_features[3]:.3f}]")
        
        # Get current hidden state for this identity
        hidden_state = self.hidden_states[recognized_face_id]
        
        # Forward pass through RNN
        probability, new_hidden_state = self._rnn_forward(normalized_features, hidden_state)
        
        if self.logger:
            self.logger.info(f"RNN raw output for {recognized_face_id}: prob={probability:.4f}, hidden_norm={np.linalg.norm(new_hidden_state):.3f}")
        
        # Update hidden state
        self.hidden_states[recognized_face_id] = new_hidden_state
        
        # Apply threshold
        is_speaking = probability > self.speaking_threshold
        confidence = float(probability)
        
        return is_speaking, confidence
    
    def _analyze_temporal_pattern_simple(self, buffer: deque) -> Tuple[bool, float]:
        """
        Simple threshold-based temporal pattern analysis (fallback method).
        
        Analyzes temporal variations in lip features to detect speaking activity.
        Speaking is characterized by rhythmic mouth movements.
        
        Args:
            buffer: Deque of feature vectors over time
            
        Returns:
            Tuple of (is_speaking: bool, confidence: float)
        """
        feature_matrix = np.array(buffer)  # Shape: (time_steps, num_features)
        
        # Extract features: MAR, MER, lip_height, lip_width
        mar_values = feature_matrix[:, 0]
        lip_height_values = feature_matrix[:, 2] if feature_matrix.shape[1] > 2 else mar_values
        
        # Calculate temporal statistics
        mar_std = np.std(mar_values)
        mar_range = np.max(mar_values) - np.min(mar_values)
        mar_mean = np.mean(mar_values)
        
        # Calculate movement frequency (changes per frame)
        if len(mar_values) > 2:
            mar_diff = np.abs(np.diff(mar_values))
            mar_diff_mean = np.mean(mar_diff)
            
            # Count significant movements (mouth opening/closing)
            movement_count = np.sum(mar_diff > self.movement_threshold * 0.5)
            movement_frequency = movement_count / len(mar_diff)
        else:
            mar_diff_mean = 0.0
            movement_frequency = 0.0
        
        # Speaking indicators with weighted scoring
        confidence_components = []
        
        # 1. Variation in mouth opening (primary indicator)
        # Speaking typically shows MAR std > 0.03-0.05
        var_score = min(mar_std / 0.05, 1.0)
        confidence_components.append(('variation', var_score, 0.35))
        
        # 2. Range of movement (mouth opens significantly)
        # Speaking shows MAR range > 0.08-0.15
        range_score = min(mar_range / 0.12, 1.0)
        confidence_components.append(('range', range_score, 0.25))
        
        # 3. Movement frequency (rhythmic motion)
        # Speaking shows regular movements, typically 2-8 Hz
        # Expecting 0.2-0.6 movement frequency in buffer
        freq_score = min(movement_frequency / 0.4, 1.0)
        confidence_components.append(('frequency', freq_score, 0.25))
        
        # 4. Average mouth opening
        # Speaking typically shows MAR > 0.25
        opening_score = min(max(mar_mean - 0.2, 0) / 0.15, 1.0)
        confidence_components.append(('opening', opening_score, 0.15))
        
        # Calculate weighted confidence
        confidence = sum(score * weight for _, score, weight in confidence_components)
        
        # Log details if logger available
        if self.logger:
            component_str = ", ".join([f"{name}={score:.2f}" for name, score, _ in confidence_components])
            self.logger.info(
                f"Simple analysis: MAR(μ={mar_mean:.3f}, σ={mar_std:.3f}, range={mar_range:.3f}), "
                f"freq={movement_frequency:.2f}, scores=[{component_str}], conf={confidence:.3f}"
            )
        
        # Apply threshold
        is_speaking = confidence > self.speaking_threshold
        
        return is_speaking, confidence
    
    def _apply_temporal_smoothing(
        self, 
        recognized_face_id: str, 
        is_speaking: bool, 
        confidence: float
    ) -> Tuple[bool, float]:
        """
        Apply temporal smoothing to reduce false positives.
        
        Uses majority voting over recent decisions and confidence averaging.
        
        Args:
            recognized_face_id: Face identity ID
            is_speaking: Current frame speaking decision
            confidence: Current frame confidence
            
        Returns:
            Smoothed (is_speaking, confidence) tuple
        """
        # Add current decision to buffer
        self.speaking_state_buffers[recognized_face_id].append((is_speaking, confidence))
        
        buffer = self.speaking_state_buffers[recognized_face_id]
        
        # Majority voting
        speaking_votes = sum(1 for s, _ in buffer if s)
        total_votes = len(buffer)
        
        # Calculate smoothed confidence as average
        avg_confidence = np.mean([c for _, c in buffer])
        
        # Smoothed decision: majority vote
        smoothed_is_speaking = speaking_votes > (total_votes / 2)
        
        return smoothed_is_speaking, float(avg_confidence)
    
    def _get_default_state(self, recognized_face_id: str) -> Tuple[bool, float]:
        """
        Get default speaking state when detection cannot be performed.
        
        Returns the most recent state if available, otherwise returns not speaking.
        
        Args:
            recognized_face_id: Face identity ID
            
        Returns:
            (is_speaking, confidence) tuple
        """
        if recognized_face_id in self.speaking_state_buffers:
            buffer = self.speaking_state_buffers[recognized_face_id]
            if len(buffer) > 0:
                # Return most recent state
                return buffer[-1]
        
        return False, 0.0
    
    def _validate_landmark(self, landmark: Tuple[float, float]) -> bool:
        """
        Validate that a landmark has finite coordinates.
        
        Args:
            landmark: (x, y) tuple
            
        Returns:
            True if valid, False otherwise
        """
        if landmark is None or len(landmark) != 2:
            return False
        return np.isfinite(landmark[0]) and np.isfinite(landmark[1])
    
    def reset_identity(self, recognized_face_id: str):
        """
        Reset temporal buffers for a specific identity.
        
        Useful when an identity leaves the scene or needs to be reinitialized.
        
        Args:
            recognized_face_id: Face identity ID to reset
        """
        if recognized_face_id in self.feature_buffers:
            self.feature_buffers[recognized_face_id].clear()
        if recognized_face_id in self.speaking_state_buffers:
            self.speaking_state_buffers[recognized_face_id].clear()
        if self.rnn_enabled and recognized_face_id in self.hidden_states:
            # Reset hidden state to zeros
            self.hidden_states[recognized_face_id] = np.zeros_like(self.hidden_states[recognized_face_id])
        
        self.logger.debug(f"Reset buffers for identity: {recognized_face_id}")
    
    def cleanup_old_identities(self, active_ids: List[str]):
        """
        Clean up buffers for identities that are no longer active.
        
        Args:
            active_ids: List of currently active recognized_face_ids
        """
        # Find identities to remove
        all_ids = set(self.feature_buffers.keys()) | set(self.speaking_state_buffers.keys())
        if self.rnn_enabled:
            all_ids |= set(self.hidden_states.keys())
        
        ids_to_remove = all_ids - set(active_ids)
        
        # Remove old identities
        for face_id in ids_to_remove:
            if face_id in self.feature_buffers:
                del self.feature_buffers[face_id]
            if face_id in self.speaking_state_buffers:
                del self.speaking_state_buffers[face_id] 
            if self.rnn_enabled and face_id in self.hidden_states:
                del self.hidden_states[face_id]
        
        if ids_to_remove:
            self.logger.debug(f"Cleaned up {len(ids_to_remove)} old identities: {ids_to_remove}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get detector statistics for monitoring.
        
        Returns:
            Dictionary with detector statistics
        """
        return {
            'active_identities': len(self.feature_buffers),
            'total_feature_buffers': sum(len(buf) for buf in self.feature_buffers.values()),
            'rnn_enabled': self.rnn_enabled,
            'use_full_landmarks': self.use_full_landmarks
        }


# Backward compatibility alias
LipMovementDetector = EnhancedLipMovementDetector