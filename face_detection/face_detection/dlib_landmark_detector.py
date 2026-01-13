#!/usr/bin/env python3
"""
Dlib Facial Landmark Detector

This module provides 68-point facial landmark detection using dlib's shape predictor.
It's designed to enhance face detection results from YOLO by adding precise facial landmarks,
particularly for lip/mouth region analysis.

The dlib 68-point landmarks follow the standard convention:
- Points 0-16: Jaw line
- Points 17-21: Right eyebrow
- Points 22-26: Left eyebrow
- Points 27-35: Nose
- Points 36-41: Right eye
- Points 42-47: Left eye
- Points 48-67: Mouth (outer and inner lip contours)

This directly maps to the ros4hri FacialLandmarks message format.
"""

import os
import dlib
import numpy as np
from typing import List, Tuple, Optional
import logging
import urllib.request
import bz2
import shutil
from pathlib import Path
    

def extract_bz2(file_path):
    # Convert string path to a Path object
    compressed_file = Path(file_path)
    
    # Define the output path (removing .bz2 extension)
    # .stem returns the filename without the last suffix
    output_file = compressed_file.with_suffix('') 

    print(f"Extracting {compressed_file.name} to {output_file.name}...")

    with bz2.open(compressed_file, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    print("Extraction complete.")

class DlibLandmarkDetector:
    """
    Detects 68 facial landmarks using dlib's shape predictor.
    
    This detector takes face bounding boxes and returns precise landmark coordinates
    that can be used for detailed facial analysis, particularly lip movement detection.
    
    Attributes:
        predictor: dlib shape_predictor object
        is_initialized: Whether the detector is ready to use
        logger: Logger for debugging
    """


    def __init__(
        self,
        model_path: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the dlib landmark detector.
        
        Args:
            model_path: Path to dlib's shape_predictor model file
                       (typically shape_predictor_68_face_landmarks.dat)
            logger: Optional logger for debugging
        """
        self.predictor = None
        self.is_initialized = False
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.default_model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        
        self._initialize(model_path)
    
    def _initialize(self, model_path: str) -> bool:
        """
        Initialize dlib shape predictor.
        
        Args:
            model_path: Path to the dlib model file
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Dlib model file not found: {model_path}")
                self.logger.info(
                    "Downloading it from: "
                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                )
                self._download_model(model_path, self.default_model_url)
                
            
            self.logger.info(f"Loading dlib shape predictor from: {model_path}")
            self.predictor = dlib.shape_predictor(model_path)
            self.is_initialized = True
            self.logger.info("Dlib landmark detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dlib landmark detector: {e}")
            self.is_initialized = False
            return False
        
    def _download_model(self, model_path: str, url: str) -> bool:
        """
        Download the dlib face detection model if it doesn't exist.
        
        Args:
            model_path: Path where to save the model
            url: URL to download the model from
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.logger.info(f"[INFO] Downloading dlib face model from {url}")

            #model path we add the zip extension
            model_path = model_path+".bz2"
            self.logger.info(f"[INFO] Saving to {model_path}")
            
            # Use a small stateful hook to log download progress without using
            # printing kwargs unsupported by ROS2 logger (e.g., end, flush).
            last_percent = -1
            def progress_hook(block_num, block_size, total_size):
                nonlocal last_percent
                # total_size can be -1 or 0 when unknown; fall back to logging bytes
                downloaded = block_num * block_size
                if total_size <= 0:
                    # Log occasionally to avoid spamming the logger
                    if block_num % 10 == 0:
                        self.logger.info(f"[INFO] Downloaded {downloaded} bytes")
                    return

                percent = min(100, (downloaded * 100) // total_size)
                # Only log when percent changes to reduce log spam
                if percent != last_percent:
                    last_percent = percent
                    self.logger.info(f"[INFO] Download progress: {percent}%")

            urllib.request.urlretrieve(url, model_path, progress_hook)
            self.logger.info(f"[INFO] Model download finished: {os.path.basename(model_path)}")
            
            #extract model from .bz2 
            extract_bz2(model_path)
            #model path without .bz2 at the end
            model_path = model_path[:-4] 

            # Verify the file was downloaded
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                self.logger.info(f"[INFO] Model downloaded successfully: {os.path.getsize(model_path)} bytes")
                return True
            else:
                self.logger.error(f"[ERROR] Downloaded file is empty or doesn't exist")
                return False
            

                
        except urllib.error.URLError as e:
            self.logger.error(f"[ERROR] Failed to download model from {url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[ERROR] Unexpected error during model download: {e}")
            return False
    
    def detect_landmarks(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Detect 68 facial landmarks for a single face.
        
        Args:
            image: Input image (BGR format, OpenCV)
            face_bbox: Face bounding box as (x, y, w, h)
            
        Returns:
            List of 68 (x, y) landmark coordinates in pixel space, or None if detection fails
        """
        if not self.is_initialized:
            return None
        
        try:
            # Convert bbox from (x, y, w, h) to dlib rectangle (left, top, right, bottom)
            x, y, w, h = face_bbox
            dlib_rect = dlib.rectangle(
                int(x),
                int(y),
                int(x + w),
                int(y + h)
            )
            
            # Detect landmarks
            shape = self.predictor(image, dlib_rect)
            
            # Convert to list of (x, y) tuples
            landmarks = []
            for i in range(68):
                point = shape.part(i)
                landmarks.append((float(point.x), float(point.y)))
            
            return landmarks
            
        except Exception as e:
            self.logger.warning(f"Failed to detect dlib landmarks: {e}")
            return None
    
    def detect_landmarks_batch(
        self,
        image: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Optional[List[Tuple[float, float]]]]:
        """
        Detect 68 facial landmarks for multiple faces.
        
        Args:
            image: Input image (BGR format, OpenCV)
            face_bboxes: List of face bounding boxes as [(x, y, w, h), ...]
            
        Returns:
            List of landmark lists (one per face), or None for failed detections
        """
        if not self.is_initialized:
            return [None] * len(face_bboxes)
        
        results = []
        for bbox in face_bboxes:
            landmarks = self.detect_landmarks(image, bbox)
            results.append(landmarks)
        
        return results
    
    @staticmethod
    def get_mouth_landmarks_indices() -> List[int]:
        """
        Get indices of mouth landmarks (points 48-67).
        
        Returns:
            List of indices for mouth landmarks
        """
        # Outer lip: 48-59
        # Inner lip: 60-67
        return list(range(48, 68))
    
    @staticmethod
    def get_lip_landmarks_for_speech_detection() -> Tuple[List[int], List[int]]:
        """
        Get indices of key lip landmarks for speech activity detection.
        
        Returns:
            Tuple of (outer_lip_indices, inner_lip_indices)
        """
        # Outer lip contour (12 points)
        outer_lip = list(range(48, 60))
        
        # Inner lip contour (8 points)
        inner_lip = list(range(60, 68))
        
        return outer_lip, inner_lip
    
    @staticmethod
    def calculate_mouth_aspect_ratio(landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) from dlib landmarks.
        
        MAR is computed as the ratio of vertical mouth opening to horizontal width.
        Higher values indicate more open mouth (speaking).
        
        Args:
            landmarks: 68-point dlib landmarks
            
        Returns:
            MAR value (float)
        """
        if len(landmarks) < 68:
            return 0.0
        
        # Vertical distances (mouth height)
        # Top to bottom of outer lip
        v1 = np.linalg.norm(
            np.array(landmarks[51]) - np.array(landmarks[57])
        )
        v2 = np.linalg.norm(
            np.array(landmarks[52]) - np.array(landmarks[56])
        )
        v3 = np.linalg.norm(
            np.array(landmarks[53]) - np.array(landmarks[55])
        )
        
        # Horizontal distance (mouth width)
        h = np.linalg.norm(
            np.array(landmarks[48]) - np.array(landmarks[54])
        )
        
        # Calculate MAR
        if h > 0:
            mar = (v1 + v2 + v3) / (3.0 * h)
        else:
            mar = 0.0
        
        return mar
    
    @staticmethod
    def get_ros4hri_landmark_mapping() -> dict:
        """
        Get mapping from dlib 68-point indices to ros4hri FacialLandmarks indices.
        
        Dlib uses 68 points (0-67), ros4hri uses 70 points (0-69) with slightly
        different indexing based on OpenPose/dlib convention.
        
        Returns:
            Dictionary mapping dlib index -> ros4hri index
        """
        # Direct mapping for most landmarks
        # Dlib and ros4hri follow the same convention for the 68 main points
        mapping = {}
        
        # Jaw line (0-16) -> same in ros4hri
        for i in range(17):
            mapping[i] = i
        
        # Eyebrows (17-26) -> same in ros4hri
        for i in range(17, 27):
            mapping[i] = i
        
        # Nose (27-35) -> same in ros4hri
        for i in range(27, 36):
            mapping[i] = i
        
        # Eyes (36-47) -> same in ros4hri
        for i in range(36, 48):
            mapping[i] = i
        
        # Mouth (48-67) -> same in ros4hri
        for i in range(48, 68):
            mapping[i] = i
        
        # Note: ros4hri has 2 additional points (68, 69) for pupils
        # which are not in standard dlib 68-point model
        
        return mapping
    
    def is_available(self) -> bool:
        """
        Check if the detector is available and ready to use.
        
        Returns:
            True if detector is initialized and ready
        """
        return self.is_initialized
