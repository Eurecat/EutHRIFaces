#!/usr/bin/env python3
"""
VSDLM (Visual Speech Detection Lightweight Model) Detector

This module provides a wrapper for PINTO0309's VSDLM model for visual speech detection.
The model analyzes mouth region crops and outputs a probability of speaking (mouth open).

Reference: https://github.com/PINTO0309/VSDLM
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import logging
import urllib.request
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None


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
    
    def __init__(
        self,
        model_path: str = "vsdlm_s.onnx",
        model_variant: str = "S",
        providers: Optional[List[str]] = None,
        crop_margin_top: int = 2,
        crop_margin_bottom: int = 8,
        crop_margin_left: int = 2,
        crop_margin_right: int = 2,
        speaking_threshold: float = 0.5,
        debug_save_crops: bool = False,
        logger: Optional[logging.Logger] = None
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
    
    def _extract_mouth_bbox(self, landmarks: List[Tuple[float, float]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract mouth bounding box from facial landmarks.
        
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
    
    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
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
        
        return batched
    
    def detect_speaking(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float]]
    ) -> Tuple[bool, float]:
        """
        Detect speaking activity from image and facial landmarks.
        
        Args:
            image: Input BGR image
            landmarks: List of (x, y) tuples for 68 facial landmarks
            
        Returns:
            Tuple of (is_speaking: bool, confidence: float)
        """
        # Extract mouth bounding box
        bbox = self._extract_mouth_bbox(landmarks)
        if bbox is None:
            if self.logger:
                self.logger.warning("Failed to extract mouth bounding box from landmarks")
            return False, 0.0
        
        x1, y1, x2, y2 = bbox
        
        if self.logger:
            self.logger.info(f"[VSDLM] Mouth bbox extracted: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}")
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            if self.logger:
                self.logger.warning(f"Invalid mouth bbox: {bbox}")
            return False, 0.0
        
        # Crop mouth region
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            if self.logger:
                self.logger.warning("Mouth crop outside image bounds")
            return False, 0.0
        
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            if self.logger:
                self.logger.warning("Empty mouth crop")
            return False, 0.0
        
        if self.logger:
            self.logger.info(f"[VSDLM] Mouth crop size: {crop.shape}, will be resized to {self.input_height}x{self.input_width}")
        
        # Debug: save crop if enabled
        if self.debug_save_crops and self.debug_frame_count % 30 == 0:  # Save every 30 frames
            try:
                debug_path = f"/tmp/vsdlm_crop_{self.debug_frame_count:06d}.jpg"
                cv2.imwrite(debug_path, crop)
                if self.logger:
                    self.logger.info(f"[VSDLM DEBUG] Saved crop to {debug_path}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to save debug crop: {e}")
        
        self.debug_frame_count += 1
        
        # Preprocess crop
        input_tensor = self._preprocess_crop(crop)
        
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
                self.logger.info(f"[VSDLM] Final result: prob_open={prob_open:.4f}, threshold={self.speaking_threshold:.4f}, is_speaking={is_speaking}")
            
            return is_speaking, prob_open
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"VSDLM inference failed: {e}", exc_info=True)
            return False, 0.0
