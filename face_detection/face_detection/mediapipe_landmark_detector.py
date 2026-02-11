#!/usr/bin/env python3
"""
MediaPipe Facial Landmark Detector

This module provides 478-point facial landmark detection using MediaPipe's Face Landmarker.
It's designed to enhance face detection results from YOLO by adding precise facial landmarks,
mapped to the ros4hri FacialLandmarks standard format (68 points + 2 pupils).

MediaPipe provides 478 3D landmarks covering the entire face mesh:
- Lips: ~40 landmarks (detailed outer and inner contours)
- Left eye: ~16 landmarks
- Right eye: ~16 landmarks  
- Left eyebrow: ~8 landmarks
- Right eyebrow: ~8 landmarks
- Face oval: ~36 landmarks
- Nose: ~15 landmarks
- And many more detailed mesh points

This module maps the most relevant MediaPipe landmarks to the 68-point dlib/ros4hri convention.

Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
Full landmark map: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
"""

import os
import numpy as np
from typing import List, Tuple, Optional
import logging
import urllib.request
from pathlib import Path

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    

class MediaPipeLandmarkDetector:
    """
    Detects 478 facial landmarks using MediaPipe Face Landmarker and maps them to 68-point ros4hri format.
    
    This detector takes face bounding boxes and returns precise landmark coordinates
    that can be used for detailed facial analysis, particularly lip movement detection.
    
    MediaPipe provides significantly more landmarks than dlib (478 vs 68), offering
    superior precision for lip and mouth tracking.
    
    Attributes:
        landmarker: MediaPipe FaceLandmarker object
        is_initialized: Whether the detector is ready to use
        logger: Logger for debugging
    """

    # MediaPipe to ros4hri/dlib landmark mapping
    # Based on verified mapping from: https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks
    # Reference: Converts MediaPipe's 478 face landmarks to Dlib's 68 face landmarks
    # Dlib indices: 0-67 (68 points total), note: the GitHub repo uses 1-68 indexing, we use 0-67
    
    MEDIAPIPE_TO_ROS4HRI = {
        # Face Contour (dlib 0-16)
        0: 127,        # dlib 0 (repo index 1)
        1: 234,        # dlib 1 (repo index 2)
        2: 93,         # dlib 2 (repo index 3)
        3: 132,        # dlib 3 (repo index 4) - note: repo averages 132 and 58
        4: 58,         # dlib 4 (repo index 5) - note: repo averages 58 and 172
        5: 136,        # dlib 5 (repo index 6)
        6: 150,        # dlib 6 (repo index 7)
        7: 176,        # dlib 7 (repo index 8)
        8: 152,        # dlib 8 (repo index 9) - chin center
        9: 400,        # dlib 9 (repo index 10)
        10: 379,       # dlib 10 (repo index 11)
        11: 365,       # dlib 11 (repo index 12)
        12: 397,       # dlib 12 (repo index 13) - note: repo averages 397 and 288
        13: 361,       # dlib 13 (repo index 14)
        14: 323,       # dlib 14 (repo index 15)
        15: 454,       # dlib 15 (repo index 16)
        16: 356,       # dlib 16 (repo index 17)
        
        # Right Eyebrow (dlib 17-21)
        17: 70,        # dlib 17 (repo index 18)
        18: 63,        # dlib 18 (repo index 19)
        19: 105,       # dlib 19 (repo index 20)
        20: 66,        # dlib 20 (repo index 21)
        21: 107,       # dlib 21 (repo index 22)
        
        # Left Eyebrow (dlib 22-26)
        22: 336,       # dlib 22 (repo index 23)
        23: 296,       # dlib 23 (repo index 24)
        24: 334,       # dlib 24 (repo index 25)
        25: 293,       # dlib 25 (repo index 26)
        26: 300,       # dlib 26 (repo index 27)
        
        # Nose (dlib 27-35)
        27: 168,       # dlib 27 (repo index 28) - note: repo averages 168 and 6
        28: 197,       # dlib 28 (repo index 29) - note: repo averages 197 and 195
        29: 5,         # dlib 29 (repo index 30)
        30: 4,         # dlib 30 (repo index 31)
        31: 75,        # dlib 31 (repo index 32)
        32: 97,        # dlib 32 (repo index 33)
        33: 2,         # dlib 33 (repo index 34)
        34: 326,       # dlib 34 (repo index 35)
        35: 305,       # dlib 35 (repo index 36)
        
        # Right Eye (dlib 36-41)
        36: 33,        # dlib 36 (repo index 37)
        37: 160,       # dlib 37 (repo index 38)
        38: 158,       # dlib 38 (repo index 39)
        39: 133,       # dlib 39 (repo index 40)
        40: 153,       # dlib 40 (repo index 41)
        41: 144,       # dlib 41 (repo index 42)
        
        # Left Eye (dlib 42-47)
        42: 362,       # dlib 42 (repo index 43)
        43: 385,       # dlib 43 (repo index 44)
        44: 387,       # dlib 44 (repo index 45)
        45: 263,       # dlib 45 (repo index 46)
        46: 373,       # dlib 46 (repo index 47)
        47: 380,       # dlib 47 (repo index 48)
        
        # Upper Lip Contour Top (dlib 48-54)
        48: 61,        # dlib 48 (repo index 49)
        49: 39,        # dlib 49 (repo index 50)
        50: 37,        # dlib 50 (repo index 51)
        51: 0,         # dlib 51 (repo index 52)
        52: 267,       # dlib 52 (repo index 53)
        53: 269,       # dlib 53 (repo index 54)
        54: 291,       # dlib 54 (repo index 55)
        
        # Lower Lip Contour Bottom (dlib 55-59)
        55: 321,       # dlib 55 (repo index 56)
        56: 314,       # dlib 56 (repo index 57)
        57: 17,        # dlib 57 (repo index 58)
        58: 84,        # dlib 58 (repo index 59)
        59: 91,        # dlib 59 (repo index 60)
        
        # Upper Lip Contour Bottom (Inner upper lip) (dlib 60-64)
        60: 78,        # dlib 60 (repo index 61)
        61: 82,        # dlib 61 (repo index 62)
        62: 13,        # dlib 62 (repo index 63)
        63: 312,       # dlib 63 (repo index 64)
        64: 308,       # dlib 64 (repo index 65)
        
        # Lower Lip Contour Top (Inner lower lip) (dlib 65-67)
        65: 317,       # dlib 65 (repo index 66)
        66: 14,        # dlib 66 (repo index 67)
        67: 87,        # dlib 67 (repo index 68)
        
        # Pupils (ros4hri 68-69) - MediaPipe iris centers
        68: 468,       # ros4hri 68: Right pupil
        69: 473,       # ros4hri 69: Left pupil
    }

    def __init__(
        self,
        model_path: str,
        logger: Optional[logging.Logger] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize the MediaPipe landmark detector.
        
        Args:
            model_path: Path to MediaPipe's face_landmarker.task model file
            logger: Optional logger for debugging
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for face tracking (0.0-1.0)
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.landmarker = None
        self.is_initialized = False
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_gpu = use_gpu
        
        # Default model URL (lightest float16 model ~3.7MB)
        self.default_model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        
        if not MEDIAPIPE_AVAILABLE:
            self.logger.error("MediaPipe is not installed. Install it with: pip install mediapipe")
            self.is_initialized = False
        else:
            self._initialize(model_path)
    
    def _initialize(self, model_path: str) -> bool:
        """
        Initialize MediaPipe Face Landmarker.
        
        Args:
            model_path: Path to the MediaPipe model file
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"MediaPipe model file not found: {model_path}")
                self.logger.info(
                    f"Downloading it from: {self.default_model_url}"
                )
                self._download_model(model_path, self.default_model_url)
            
            self.logger.info(f"Loading MediaPipe Face Landmarker from: {model_path}")
            
            # Create MediaPipe FaceLandmarker options
            base_options = python.BaseOptions(model_asset_path=model_path)
            
            # Configure GPU delegation if requested
            if self.use_gpu:
                try:
                    # Try to enable GPU delegation
                    from mediapipe.tasks.python.core import base_options as bo
                    base_options.delegate = bo.BaseOptions.Delegate.GPU
                    light_green = "\033[38;5;82m"
                    reset = "\033[0m"
                    self.logger.info(f"{light_green}[MEDIAPIPE-GPU] GPU delegation enabled{reset}")
                except Exception as gpu_e:
                    self.logger.warning(f"[MEDIAPIPE-GPU] Failed to enable GPU delegation: {gpu_e}")
                    self.logger.info("[MEDIAPIPE-GPU] Falling back to CPU")
            else:
                light_green = "\033[38;5;82m"
                reset = "\033[0m"
                self.logger.info(f"{light_green}[MEDIAPIPE-CPU] Using CPU{reset}")
            
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=10,  # Support multiple faces
                min_face_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=False,  # We don't need blendshapes
                output_facial_transformation_matrixes=False  # We don't need transformation matrices
            )
            
            # Create the landmarker
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.is_initialized = True
            self.logger.info("MediaPipe landmark detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe landmark detector: {e}")
            self.is_initialized = False
            return False
        
    def _download_model(self, model_path: str, url: str) -> bool:
        """
        Download the MediaPipe face landmarker model if it doesn't exist.
        
        Args:
            model_path: Path where to save the model
            url: URL to download the model from
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.logger.info(f"[INFO] Downloading MediaPipe face landmarker model from {url}")
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
        Detect facial landmarks for a single face using MediaPipe, mapped to ros4hri 68-point format.
        
        Note: MediaPipe detects faces internally, but we crop to the provided bbox
        for consistency with YOLO detections.
        
        Args:
            image: Input image (BGR format, OpenCV)
            face_bbox: Face bounding box as (x, y, w, h)
            
        Returns:
            List of 68 (x, y) landmark coordinates in pixel space (ros4hri format), or None if detection fails
        """
        if not self.is_initialized:
            return None
        
        try:
            # Crop image to face bbox to focus MediaPipe detection
            x, y, w, h = face_bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure bbox is within image bounds
            img_h, img_w = image.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            
            if x2 <= x1 or y2 <= y1:
                self.logger.warning(f"Invalid face bbox: {face_bbox}")
                return None
            
            face_crop = image[y1:y2, x1:x2]
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_image = face_crop[:, :, ::-1].copy()
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect landmarks
            detection_result = self.landmarker.detect(mp_image)
            
            if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                return None
            
            # Get first face's landmarks (we cropped to one face)
            mediapipe_landmarks = detection_result.face_landmarks[0]
            
            # Convert MediaPipe normalized landmarks to pixel coordinates in cropped space
            crop_h, crop_w = face_crop.shape[:2]
            
            # Map MediaPipe 478 landmarks to ros4hri 68 landmarks + 2 pupils
            ros4hri_landmarks = []
            for ros4hri_idx in range(70):  # 68 main + 2 pupils
                if ros4hri_idx in self.MEDIAPIPE_TO_ROS4HRI:
                    mp_idx = self.MEDIAPIPE_TO_ROS4HRI[ros4hri_idx]
                    if mp_idx < len(mediapipe_landmarks):
                        landmark = mediapipe_landmarks[mp_idx]
                        # Convert from normalized coordinates to pixel coordinates in cropped image
                        px = landmark.x * crop_w
                        py = landmark.y * crop_h
                        # Transform back to original image coordinates
                        px_global = px + x1
                        py_global = py + y1
                        ros4hri_landmarks.append((float(px_global), float(py_global)))
                    else:
                        # Fallback if index out of range
                        ros4hri_landmarks.append((0.0, 0.0))
                else:
                    # If no mapping exists, use (0, 0) placeholder
                    ros4hri_landmarks.append((0.0, 0.0))
            
            # Return only the first 68 landmarks (standard ros4hri, pupils handled separately)
            return ros4hri_landmarks[:68]
            
        except Exception as e:
            self.logger.warning(f"Failed to detect MediaPipe landmarks: {e}")
            return None
    
    def detect_landmarks_batch(
        self,
        image: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Optional[List[Tuple[float, float]]]]:
        """
        Detect facial landmarks for multiple faces.
        
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
    
    def get_pupils(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get pupil coordinates using MediaPipe iris landmarks.
        
        Args:
            image: Input image (BGR format, OpenCV)
            face_bbox: Face bounding box as (x, y, w, h)
            
        Returns:
            Tuple of (right_pupil, left_pupil) coordinates, or None if detection fails
        """
        if not self.is_initialized:
            return None
        
        try:
            # Use the same detection as landmarks
            x, y, w, h = face_bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            img_h, img_w = image.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            face_crop = image[y1:y2, x1:x2]
            rgb_image = face_crop[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            detection_result = self.landmarker.detect(mp_image)
            
            if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                return None
            
            mediapipe_landmarks = detection_result.face_landmarks[0]
            crop_h, crop_w = face_crop.shape[:2]
            
            # Right pupil (MediaPipe index 468)
            right_pupil_mp = mediapipe_landmarks[468]
            right_pupil = (
                float(right_pupil_mp.x * crop_w + x1),
                float(right_pupil_mp.y * crop_h + y1)
            )
            
            # Left pupil (MediaPipe index 473)
            left_pupil_mp = mediapipe_landmarks[473]
            left_pupil = (
                float(left_pupil_mp.x * crop_w + x1),
                float(left_pupil_mp.y * crop_h + y1)
            )
            
            return (right_pupil, left_pupil)
            
        except Exception as e:
            self.logger.warning(f"Failed to detect pupils: {e}")
            return None
    
    @staticmethod
    def calculate_mouth_aspect_ratio(landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) from ros4hri landmarks.
        
        MAR is computed as the ratio of vertical mouth opening to horizontal width.
        Higher values indicate more open mouth (speaking).
        
        Args:
            landmarks: 68-point ros4hri landmarks
            
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
    
    def is_available(self) -> bool:
        """
        Check if the detector is available and ready to use.
        
        Returns:
            True if detector is initialized and ready
        """
        return self.is_initialized
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except:
                pass
