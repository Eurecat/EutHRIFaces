#!/usr/bin/env python3
"""
VSDLM (Visual Speech Detection Lightweight Model) Detector

This module provides a wrapper for PINTO0309's VSDLM model for visual speech detection.
The model analyzes mouth region crops and outputs a probability of speaking (mouth open).

Reference: https://github.com/PINTO0309/VSDLM
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import logging
import urllib.request
import cv2
from collections import deque, defaultdict

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_MEAN = [0, 0, 0]
DEFAULT_STD = [1, 1, 1]

# DEFAULT_MEAN = [0.485, 0.456, 0.406]
# DEFAULT_STD = [0.229, 0.224, 0.225]

class VSDLMDetector:
    """
    Visual Speech Detection using PINTO0309's VSDLM ONNX model.
    
    This detector:
    1. Takes facial landmarks (68-point dlib format)
    2. Extracts and crops the mouth region from the image
    3. Runs VSDLM inference to get speaking probability
    4. Returns binary speaking status and confidence
    """
    
    # VSDLM model variants with download URLs
    VSDLM_MODELS = {
        'P': {
            'url': 'https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_p.onnx',
            'size': '112 KB',
            'f1': 0.9502,
            'latency_ms': 0.18
        },
        'N': {
            'url': 'https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_n.onnx',
            'size': '176 KB',
            'f1': 0.9586,
            'latency_ms': 0.31
        },
        'S': {
            'url': 'https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_s.onnx',
            'size': '494 KB',
            'f1': 0.9696,
            'latency_ms': 0.50
        },
        'M': {
            'url': 'https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_m.onnx',
            'size': '1.7 MB',
            'f1': 0.9801,
            'latency_ms': 0.70
        },
        'L': {
            'url': 'https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_l.onnx',
            'size': '6.4 MB',
            'f1': 0.9891,
            'latency_ms': 0.91
        }
    }
    
    # Mouth landmark indices (dlib 68-point convention)
    MOUTH_OUTER_INDICES = list(range(48, 60))  # Outer lip contour (12 points)
    MOUTH_INNER_INDICES = list(range(60, 68))  # Inner lip contour (8 points)
    
    # YOLO 5-point landmark indices (when using YOLO face detection)
    # YOLO provides: left_eye, right_eye, nose, left_mouth, right_mouth
    # In the 68-point array, YOLO landmarks appear at specific indices with c > 0
    YOLO_LEFT_MOUTH_IDX = 48   # Left mouth corner in YOLO 5-point (index 3)
    YOLO_RIGHT_MOUTH_IDX = 54  # Right mouth corner in YOLO 5-point (index 4)
    
    def __init__(
        self,
        model_path: str = "vsdlm_s.onnx",
        model_variant: str = "S",
        providers: Optional[List[str]] = None,
        crop_margin_top: int = 0,
        crop_margin_bottom: int = 0,
        crop_margin_left: int = 0,
        crop_margin_right: int = 0,
        speaking_threshold: float = 0.5,
        debug_save_crops: bool = False,
        logger: Optional[logging.Logger] = None,
        mouth_height_ratio: float = 0.35,
        temporal_smoothing: bool = True,
        smoothing_window_size: int = 5,
        min_confidence_for_change: float = 0.1
    ):
        """
        Initialize VSDLM detector.
        
        Args:
            model_path: Path to VSDLM ONNX model file
            model_variant: Model size variant ('P', 'N', 'S', 'M', 'L') for auto-download
            providers: ONNX execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            crop_margin_top: Top margin pixels when cropping mouth region
            crop_margin_bottom: Bottom margin pixels when cropping mouth region
            crop_margin_left: Left margin pixels when cropping mouth region
            crop_margin_right: Right margin pixels when cropping mouth region
            speaking_threshold: Probability threshold for classifying as speaking (default: 0.5)
            debug_save_crops: Save mouth crops to /tmp for debugging
            logger: Optional logger for debugging
            mouth_height_ratio: Ratio of face height to use for mouth crop height (YOLO mode, default: 0.35)
            temporal_smoothing: Enable temporal smoothing to reduce flickering (default: True)
            smoothing_window_size: Number of frames to use for temporal smoothing (default: 5)
            min_confidence_for_change: Minimum confidence difference required to change state (default: 0.1)
        """
        if ort is None:
            raise ImportError("onnxruntime is required for VSDLM. Install with: pip install onnxruntime")
        
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.model_variant = model_variant.upper()
        self.crop_margin_top = crop_margin_top
        self.crop_margin_bottom = crop_margin_bottom
        self.crop_margin_left = crop_margin_left
        self.crop_margin_right = crop_margin_right
        self.speaking_threshold = speaking_threshold
        self.debug_save_crops = debug_save_crops
        self.debug_frame_count = 0
        self.mouth_height_ratio = mouth_height_ratio
        
        # Temporal smoothing parameters
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window_size = smoothing_window_size
        self.min_confidence_for_change = min_confidence_for_change
        
        # Per-identity temporal buffers for smoothing
        # Dictionary mapping face_id -> deque of (is_speaking, confidence) tuples
        self.speaking_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window_size)
        )
        
        # Last stable state per identity
        self.last_stable_state: Dict[str, Tuple[bool, float]] = {}
        
        # Auto-download model if not found
        if not self.model_path.exists():
            self._download_model()
        
        # Initialize ONNX session
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.logger.info(f"Loading VSDLM model from {self.model_path} with providers: {providers}")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        # Get model input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape

        #get the model mean and std
        model_meta = self.session.get_modelmeta()

        self.logger.info(f"[VSDLM] Model metadata map: {model_meta.custom_metadata_map}")

        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD

        if model_meta.custom_metadata_map:
            self.mean = model_meta.custom_metadata_map.get("mean", DEFAULT_MEAN)
            self.std = model_meta.custom_metadata_map.get("std", DEFAULT_STD)

        self.logger.info(f"[VSDLM] Model loaded successfully")
        self.logger.info(f"[VSDLM] Input: name='{self.input_name}', shape={input_shape}")
        self.logger.info(f"[VSDLM] Output: name='{self.output_name}', shape={output_shape}")
        
        # Extract input dimensions (handle dynamic batch dimension)
        if len(input_shape) == 4:  # [batch, channels, height, width]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
        else:
            # Default fallback
            self.input_height = 30
            self.input_width = 48
        
        self.logger.info(f"[VSDLM] Initialized: input_size={self.input_height}x{self.input_width}, speaking_threshold={self.speaking_threshold}")
        self.logger.info(f"[VSDLM] Temporal smoothing: {'enabled' if self.temporal_smoothing else 'disabled'}, window_size={self.smoothing_window_size}")
    
    def _apply_temporal_smoothing(
        self,
        face_id: str,
        is_speaking: bool,
        confidence: float
    ) -> Tuple[bool, float]:
        """
        Apply temporal smoothing to reduce speaking detection flickering.
        
        Uses a sliding window of recent predictions with majority voting
        and confidence averaging to produce stable results.
        
        Args:
            face_id: Face identifier for maintaining per-identity history
            is_speaking: Current frame speaking detection
            confidence: Current frame confidence score
            
        Returns:
            Tuple of (smoothed_is_speaking, smoothed_confidence)
        """
        # Add current prediction to history
        self.speaking_history[face_id].append((is_speaking, confidence))
        
        history = self.speaking_history[face_id]
        
        # If we don't have enough history, return current prediction
        if len(history) < 2:
            self.last_stable_state[face_id] = (is_speaking, confidence)
            return is_speaking, confidence
        
        # Calculate statistics from recent history
        recent_speaking = [s for s, c in history]
        recent_confidences = [c for s, c in history]
        
        # Majority voting for speaking decision
        speaking_votes = sum(recent_speaking)
        total_votes = len(recent_speaking)
        speaking_ratio = speaking_votes / total_votes
        
        # Average confidence
        avg_confidence = np.mean(recent_confidences)
        
        # Get last stable state
        last_speaking, last_confidence = self.last_stable_state.get(face_id, (False, 0.0))
        
        # Determine new stable state with hysteresis
        if speaking_ratio >= 0.6:  # 60% or more votes for speaking
            new_speaking = True
        elif speaking_ratio <= 0.4:  # 40% or fewer votes for speaking
            new_speaking = False
        else:
            # Inconclusive - check confidence difference
            confidence_diff = abs(avg_confidence - last_confidence)
            if confidence_diff > self.min_confidence_for_change:
                # Strong enough signal to change state
                new_speaking = avg_confidence >= self.speaking_threshold
            else:
                # Keep previous state (hysteresis)
                new_speaking = last_speaking
        
        # Update stable state
        self.last_stable_state[face_id] = (new_speaking, avg_confidence)
        
        if self.logger:
            self.logger.info(
                f"[VSDLM-SMOOTH] Face {face_id}: votes={speaking_votes}/{total_votes} "
                f"({speaking_ratio:.2f}), avg_conf={avg_confidence:.3f}, "
                f"raw={is_speaking}, smoothed={new_speaking}"
            )
        
        return new_speaking, avg_confidence
    
    def reset_identity(self, face_id: str):
        """
        Reset temporal buffers for a specific face identity.
        
        Useful when a face leaves the scene or tracking is reset.
        
        Args:
            face_id: Face identifier to reset
        """
        if face_id in self.speaking_history:
            del self.speaking_history[face_id]
        if face_id in self.last_stable_state:
            del self.last_stable_state[face_id]
        
        if self.logger:
            self.logger.debug(f"[VSDLM] Reset temporal buffers for face {face_id}")
    
    def cleanup_old_identities(self, active_face_ids: List[str]):
        """
        Clean up temporal buffers for faces that are no longer active.
        
        Args:
            active_face_ids: List of currently active face IDs
        """
        active_set = set(active_face_ids)
        
        # Find identities to remove
        history_ids = set(self.speaking_history.keys())
        state_ids = set(self.last_stable_state.keys())
        all_ids = history_ids | state_ids
        
        ids_to_remove = all_ids - active_set
        
        # Remove old identities
        for face_id in ids_to_remove:
            if face_id in self.speaking_history:
                del self.speaking_history[face_id]
            if face_id in self.last_stable_state:
                del self.last_stable_state[face_id]
        
        if ids_to_remove and self.logger:
            self.logger.debug(f"[VSDLM] Cleaned up temporal buffers for {len(ids_to_remove)} old identities")
    
    def _download_model(self):
        """Download VSDLM model if not present."""
        if self.model_variant not in self.VSDLM_MODELS:
            raise ValueError(f"Invalid model variant '{self.model_variant}'. Choose from: {list(self.VSDLM_MODELS.keys())}")
        
        model_info = self.VSDLM_MODELS[self.model_variant]
        url = model_info['url']
        
        self.logger.info(f"Downloading VSDLM model variant '{self.model_variant}' ({model_info['size']}, F1={model_info['f1']:.4f})...")
        self.logger.info(f"URL: {url}")
        self.logger.info(f"Saving to: {self.model_path}")
        
        # Create directory if needed
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                if block_num % 10 == 0:  # Log every 10 blocks to reduce verbosity
                    self.logger.info(f"Download progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
        
        try:
            urllib.request.urlretrieve(url, self.model_path, reporthook=reporthook)
            self.logger.info(f"Successfully downloaded VSDLM model to {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download VSDLM model: {e}")
    
    def _detect_landmark_type(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        Detect whether we have dlib 68-point landmarks or YOLO 5-point landmarks.
        
        Args:
            landmarks: List of (x, y, c) tuples where c is confidence
            
        Returns:
            'dlib68' or 'yolo5'
        """
        if len(landmarks) < 68:
            return 'yolo5'
        
        # Count landmarks with confidence > 0
        valid_landmarks = sum(1 for lm in landmarks if len(lm) >= 3 and lm[2] > 0.0)
        
        # If we have 5-7 valid landmarks, it's YOLO 5-point
        # If we have more than 60 valid landmarks, it's dlib 68-point
        if valid_landmarks < 10:
            return 'yolo5'
        else:
            return 'dlib68'
    
    def _extract_mouth_bbox_dlib(self, landmarks: List[Tuple[float, float]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract mouth bounding box from dlib 68-point facial landmarks.
        
        Args:
            landmarks: List of (x, y) tuples for all 68 facial landmarks
            
        Returns:
            Tuple of (x1, y1, x2, y2) or None if extraction fails
        """
        if len(landmarks) < 68:
            return None
        
        # Get all mouth landmarks (outer + inner)
        all_mouth_indices = self.MOUTH_OUTER_INDICES + self.MOUTH_INNER_INDICES
        mouth_points = [landmarks[i] for i in all_mouth_indices]
        
        # Calculate bounding box
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]
        
        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))
        
        # Apply margins
        x1 = max(0, x1 - self.crop_margin_left)
        y1 = max(0, y1 - self.crop_margin_top)
        x2 = x2 + self.crop_margin_right
        y2 = y2 + self.crop_margin_bottom
        
        return (x1, y1, x2, y2)
    
    def _extract_mouth_bbox_yolo(
        self, 
        landmarks: List[Tuple[float, float, float]], 
        face_bbox: Tuple[float, float, float, float],
        image_width: int,
        image_height: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract mouth bounding box from YOLO 5-point landmarks and face bbox.
        
        YOLO provides only 5 landmarks, with mouth corners at indices 48 and 54.
        The mouth WIDTH is determined directly from the distance between the two mouth landmarks.
        The mouth HEIGHT is estimated as a ratio of the face height.
        
        Args:
            landmarks: List of (x, y, c) tuples (68-point array but only 5 are valid)
            face_bbox: Face bounding box (xmin, ymin, xmax, ymax) in normalized coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates or None if extraction fails
        """
        # Try to get mouth corner landmarks from YOLO
        left_mouth = None
        right_mouth = None
        
        if len(landmarks) > self.YOLO_LEFT_MOUTH_IDX and landmarks[self.YOLO_LEFT_MOUTH_IDX][2] > 0:
            left_mouth = (landmarks[self.YOLO_LEFT_MOUTH_IDX][0], landmarks[self.YOLO_LEFT_MOUTH_IDX][1])
        
        if len(landmarks) > self.YOLO_RIGHT_MOUTH_IDX and landmarks[self.YOLO_RIGHT_MOUTH_IDX][2] > 0:
            right_mouth = (landmarks[self.YOLO_RIGHT_MOUTH_IDX][0], landmarks[self.YOLO_RIGHT_MOUTH_IDX][1])
        
        # We MUST have both mouth corners for YOLO mode
        if left_mouth is None or right_mouth is None:
            if self.logger:
                self.logger.warning("[VSDLM-YOLO] Missing mouth corner landmarks - cannot extract mouth bbox")
            return None
        
        # Calculate mouth center and width from the landmarks
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        
        # IMPORTANT: Mouth WIDTH comes directly from the landmark distance
        # This is the EXACT distance between the two mouth corners - DO NOT EXPAND
        mouth_width = abs(right_mouth[0] - left_mouth[0])
        
        # Convert face bbox from normalized to pixel coordinates for height estimation
        face_ymin = int(face_bbox[1] * image_height)
        face_ymax = int(face_bbox[3] * image_height)
        face_height = face_ymax - face_ymin
        
        if face_height <= 0:
            if self.logger:
                self.logger.warning("[VSDLM-YOLO] Invalid face height")
            return None
        
        # Mouth HEIGHT is estimated as a ratio of face height
        # Use the ratio directly as configured - no additional expansion
        mouth_height = face_height * self.mouth_height_ratio
        
        # Calculate crop bbox using the EXACT landmark width and estimated height
        x1 = int(mouth_center_x - mouth_width / 2)
        y1 = int(mouth_center_y - mouth_height / 2)
        x2 = int(mouth_center_x + mouth_width / 2)
        y2 = int(mouth_center_y + mouth_height / 2)
        
        # Apply additional margins (same as dlib mode)
        x1 = max(0, x1 - self.crop_margin_left)
        y1 = max(0, y1 - self.crop_margin_top)
        x2 = min(image_width, x2 + self.crop_margin_right)
        y2 = min(image_height, y2 + self.crop_margin_bottom)
        
        if self.logger:
            self.logger.info(
                f"[VSDLM-YOLO] Mouth bbox from landmarks: "
                f"left_mouth=({left_mouth[0]:.1f},{left_mouth[1]:.1f}), right_mouth=({right_mouth[0]:.1f},{right_mouth[1]:.1f}), "
                f"mouth_width={mouth_width:.1f} (EXACT landmark distance), "
                f"mouth_height={mouth_height:.1f} (face_height={face_height} * {self.mouth_height_ratio}), "
                f"mouth_crop=({x1},{y1},{x2},{y2}), final_size={x2-x1}x{y2-y1}"
            )
        
        # Store the bbox for visualization
        return (x1, y1, x2, y2)
    
    def _extract_mouth_bbox(self, landmarks: List[Tuple[float, float]]) -> Optional[Tuple[int, int, int, int]]:
        """
        DEPRECATED: Legacy method for backward compatibility.
        Use _extract_mouth_bbox_dlib() instead.
        
        Extract mouth bounding box from facial landmarks.
        
        Args:
            landmarks: List of (x, y) tuples for all 68 facial landmarks
            
        Returns:
            Tuple of (x1, y1, x2, y2) or None if extraction fails
        """
        return self._extract_mouth_bbox_dlib(landmarks)

    def _preprocess_crop(self, crop: np.ndarray, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> np.ndarray:
        """
        Preprocess mouth crop for VSDLM inference.
        
        Args:
            crop: BGR mouth region image
            
        Returns:
            Preprocessed tensor ready for inference [1, 3, H, W]
        """
       
        # Resize to model input size
        resized = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format: (H, W, C) -> (C, H, W)
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        batched = np.expand_dims(transposed, axis=0)


        # Normalize(mean, std)
        if mean is None:
            mean = DEFAULT_MEAN
        if std is None:
            std = DEFAULT_STD

        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std_arr  = np.array(std,  dtype=np.float32).reshape(1, 3, 1, 1)

        batched = (batched - mean_arr) / std_arr

        return batched
    
    def detect_speaking(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float]],
        face_bbox: Optional[Tuple[float, float, float, float]] = None,
        face_id: Optional[str] = None
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """
        Detect speaking activity from image and facial landmarks.
        
        Supports both dlib 68-point landmarks and YOLO 5-point landmarks.
        For YOLO landmarks, face_bbox is required for mouth region estimation.
        
        Args:
            image: Input BGR image
            landmarks: List of (x, y) or (x, y, c) tuples for facial landmarks
            face_bbox: Face bounding box (xmin, ymin, xmax, ymax) in normalized coords (required for YOLO)
            face_id: Face identifier for temporal smoothing (optional, improves stability)
            
        Returns:
            Tuple of (is_speaking: bool, confidence: float, mouth_bbox: Optional[Tuple[int, int, int, int]])
            mouth_bbox is (x1, y1, x2, y2) in pixel coordinates for visualization
        """
        h, w = image.shape[:2]
        
        # Detect landmark type
        landmark_type = self._detect_landmark_type(landmarks)
        
        if self.logger:
            self.logger.info(f"[VSDLM] Detected landmark type: {landmark_type}, landmark count: {len(landmarks)}")
        
        # Extract mouth bounding box based on landmark type
        if landmark_type == 'dlib68':
            # Convert to simple (x, y) tuples if needed
            simple_landmarks = [(lm[0], lm[1]) for lm in landmarks]
            bbox = self._extract_mouth_bbox_dlib(simple_landmarks)
        else:  # yolo5
            if face_bbox is None:
                if self.logger:
                    self.logger.warning("YOLO landmarks detected but no face_bbox provided")
                return False, 0.0, None
            bbox = self._extract_mouth_bbox_yolo(landmarks, face_bbox, w, h)
        
        if bbox is None:
            if self.logger:
                self.logger.warning(f"Failed to extract mouth bounding box from {landmark_type} landmarks")
            return False, 0.0, None
        
        x1, y1, x2, y2 = bbox
        
        if self.logger:
            self.logger.info(f"[VSDLM] Mouth bbox extracted: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}")
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            if self.logger:
                self.logger.warning(f"Invalid mouth bbox: {bbox}")
            return False, 0.0, None
        
        # Crop mouth region
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            if self.logger:
                self.logger.warning("Mouth crop outside image bounds")
            return False, 0.0, None
        
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            if self.logger:
                self.logger.warning("Empty mouth crop")
            return False, 0.0, None
        
        if self.logger:
            self.logger.info(f"[VSDLM] Mouth crop size: {crop.shape}, will be resized to {self.input_height}x{self.input_width}")
        
        # Debug: save crop if enabled
        if self.debug_save_crops and self.debug_frame_count % 30 == 0:  # Save every 30 frames
            try:
                debug_path = f"/tmp/.X11-unix/vsdlm_crop_{self.debug_frame_count:06d}.jpg"
                resized = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(debug_path, resized)
                if self.logger:
                    self.logger.info(f"[VSDLM DEBUG] Saved crop to {debug_path}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to save debug crop: {e}")
        
        self.debug_frame_count += 1
        
        # Preprocess crop
        input_tensor = self._preprocess_crop(crop, mean=self.mean, std=self.std)
        
        if self.logger:
            self.logger.info(f"[VSDLM] Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}, range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        
        # Run inference
        try:
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            if self.logger:
                self.logger.info(f"[VSDLM] Raw output: {outputs}, type: {type(outputs)}")
                self.logger.info(f"[VSDLM] Output[0]: {outputs[0]}, shape: {outputs[0].shape if hasattr(outputs[0], 'shape') else 'N/A'}")
            
            # Extract probability - check output format
            raw_output = outputs[0]
            if hasattr(raw_output, 'shape'):
                if len(raw_output.shape) == 0:
                    # Scalar output
                    prob_open = float(raw_output)
                elif raw_output.shape == (1,):
                    # Single element array
                    prob_open = float(raw_output[0])
                elif len(raw_output.shape) == 2 and raw_output.shape[0] == 1:
                    # Batch output [1, 1]
                    prob_open = float(raw_output[0][0])
                else:
                    if self.logger:
                        self.logger.error(f"Unexpected output shape: {raw_output.shape}")
                    prob_open = float(raw_output.flatten()[0])
            else:
                prob_open = float(raw_output)
            
            # Classify as speaking if probability above threshold
            is_speaking = prob_open >= self.speaking_threshold
            
            if self.logger:
                self.logger.info(f"[VSDLM] Raw result: prob_open={prob_open:.4f}, threshold={self.speaking_threshold:.4f}, is_speaking={is_speaking}")
            
            # Apply temporal smoothing if enabled and face_id is provided
            self.logger.debug(f"[VSDLM] Temporal smoothing: {'enabled' if self.temporal_smoothing else 'disabled'}")
            self.logger.debug(f"[VSDLM] face_id: {face_id}") 
            if self.temporal_smoothing and face_id is not None:
                self.logger.info(f"[VSDLM] Applying temporal smoothing for face_id: {face_id}")
                smoothed_is_speaking, smoothed_confidence = self._apply_temporal_smoothing(
                    face_id, is_speaking, prob_open
                )
                self.logger.info(f"[VSDLM] Smoothed result: is_speaking={smoothed_is_speaking}, confidence={smoothed_confidence:.4f}")  
                if self.logger:
                    self.logger.info(
                        f"[VSDLM] Temporal smoothing applied for face {face_id}: "
                        f"raw=({is_speaking}, {prob_open:.3f}) -> smoothed=({smoothed_is_speaking}, {smoothed_confidence:.3f})"
                    )
                
                # Use smoothed results
                final_is_speaking = smoothed_is_speaking
                final_confidence = smoothed_confidence
            else:
                # Use raw results
                final_is_speaking = is_speaking
                final_confidence = prob_open
                
                if self.temporal_smoothing and face_id is None and self.logger:
                    self.logger.info("[VSDLM] Temporal smoothing disabled: no face_id provided")
            
            if self.logger:
                self.logger.info(f"[VSDLM] Final result: is_speaking={final_is_speaking}, confidence={final_confidence:.4f}")
            
            return final_is_speaking, final_confidence, bbox
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"VSDLM inference failed: {e}", exc_info=True)
            return False, 0.0, None
