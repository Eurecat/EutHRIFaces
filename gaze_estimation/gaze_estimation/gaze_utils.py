"""
Gaze computation utilities for facial landmark analysis.

This module contains helper functions for computing gaze direction
and scores from facial landmarks using computer vision techniques.
"""

import numpy as np
import cv2
from math import degrees, sqrt, atan2
from typing import Tuple, Optional, List


class GazeComputer:
    """
    Class for computing gaze direction and score from facial landmarks.
    
    Uses a pinhole camera model and PnP (Perspective-n-Point) algorithm
    to estimate head pose and derive gaze information.
    """
    
    def __init__(self, 
                 focal_length: float,
                 center_x: float, 
                 center_y: float,
                 max_angle_threshold: float = 80.0,
                 method: str = "pnp"):
        """
        Initialize the gaze computer.
        
        Args:
            focal_length: Camera focal length
            center_x: Camera principal point X coordinate
            center_y: Camera principal point Y coordinate  
            max_angle_threshold: Maximum angle for "fully looking away" in degrees
            method: Gaze computation method - "eye_vector" (stable) or "pnp" (original)
        """
        self.focal_length = focal_length
        self.center_x = center_x
        self.center_y = center_y
        self.max_angle_threshold = max_angle_threshold
        self.method = method
        
        # Setup camera matrix
        self.camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype="double")
        
        # No lens distortion assumed
        self.dist_coeffs = np.zeros((4, 1))
        
        # 3D face model points (generic face model in mm) - only needed for PnP method
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (20.0, -30.0, -20.0),        # Right eye
            (-20.0, -30.0, -20.0),       # Left eye  
            (20.0, 30.0, -20.0),         # Right lip corner
            (-20.0, 30.0, -20.0),        # Left lip corner
            (0.0, 30.0, -20.0)           # Mouth center
        ], dtype="double")
    
    def compute_gaze(self, 
                    nose: Tuple[float, float],
                    right_eye: Tuple[float, float], 
                    left_eye: Tuple[float, float],
                    right_lip: Tuple[float, float],
                    left_lip: Tuple[float, float]) -> Optional[Tuple[float, np.ndarray, float, float, float]]:
        """
        Compute gaze score and direction from 5 key facial landmarks.
        
        Args:
            nose: Nose tip coordinates (x, y)
            right_eye: Right eye coordinates (x, y)
            left_eye: Left eye coordinates (x, y) 
            right_lip: Right lip corner coordinates (x, y)
            left_lip: Left lip corner coordinates (x, y)
            
        Returns:
            Tuple of (gaze_score, gaze_direction_3d, pitch, yaw, roll) or None if computation fails
        """
        if self.method == "eye_vector":
            return self._compute_gaze_eye_vector_method(nose, right_eye, left_eye, right_lip, left_lip)
        elif self.method == "pnp":
            return self._compute_gaze_pnp_method(nose, right_eye, left_eye, right_lip, left_lip)
        else:
            raise ValueError(f"Unknown gaze computation method: {self.method}")
    
    def _compute_gaze_eye_vector_method(self, nose, right_eye, left_eye, right_lip, left_lip):
        """
        Robust eye vector based gaze estimation - more stable than PnP.
        """
        # Eye center point
        eye_center = ((right_eye[0] + left_eye[0]) / 2.0, (right_eye[1] + left_eye[1]) / 2.0)
        
        # Mouth center
        mouth_center = ((right_lip[0] + left_lip[0]) / 2.0, (right_lip[1] + left_lip[1]) / 2.0)
        
        # Face center (between eyes and mouth)
        face_center = ((eye_center[0] + mouth_center[0]) / 2.0, (eye_center[1] + mouth_center[1]) / 2.0)
        
        # Convert to camera coordinates (subtract principal point)
        eye_cam_x = eye_center[0] - self.center_x
        eye_cam_y = eye_center[1] - self.center_y
        nose_cam_x = nose[0] - self.center_x
        nose_cam_y = nose[1] - self.center_y
        
        # Estimate head yaw from eye-nose relationship
        # Positive yaw = head turned right
        eye_nose_diff_x = nose_cam_x - eye_cam_x
        yaw_rad = atan2(eye_nose_diff_x, self.focal_length * 0.5)  # Scale factor for sensitivity
        yaw_deg = degrees(yaw_rad)
        
        # Estimate head pitch from vertical face position
        # Positive pitch = head tilted up
        face_cam_y = face_center[1] - self.center_y
        pitch_rad = atan2(-face_cam_y, self.focal_length * 0.8)  # Scale factor for sensitivity
        pitch_deg = degrees(pitch_rad)
        
        # Estimate roll from eye line tilt
        eye_diff_x = left_eye[0] - right_eye[0]
        eye_diff_y = left_eye[1] - right_eye[1]
        roll_rad = atan2(eye_diff_y, eye_diff_x)
        roll_deg = degrees(roll_rad)
        
        # Compute gaze score based on angles
        yaw_score = max(0.0, 1.0 - abs(yaw_deg) / self.max_angle_threshold)
        pitch_score = max(0.0, 1.0 - abs(pitch_deg) / self.max_angle_threshold)
        final_score = (yaw_score + pitch_score) / 2.0
        
        # Create 3D gaze direction vector
        # Convert angles to direction vector
        gaze_direction_3d = np.array([
            np.sin(yaw_rad),
            -np.sin(pitch_rad),
            np.cos(yaw_rad) * np.cos(pitch_rad)
        ])
        # Normalize
        gaze_direction_3d = gaze_direction_3d / np.linalg.norm(gaze_direction_3d)
        
        return final_score, gaze_direction_3d, pitch_deg, yaw_deg, roll_deg
    
    def _compute_gaze_pnp_method(self, nose, right_eye, left_eye, right_lip, left_lip):
        """
        Original PnP-based method (kept for comparison).
        """
        # Compute mouth center
        mouth_center = (
            (right_lip[0] + left_lip[0]) / 2.0,
            (right_lip[1] + left_lip[1]) / 2.0
        )
        
        # Create image points array for solvePnP
        image_points = np.array([
            nose,         # Nose tip
            right_eye,    # Right eye
            left_eye,     # Left eye
            right_lip,    # Right lip corner
            left_lip,     # Left lip corner
            mouth_center  # Mouth center
        ], dtype="double")
        
        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_SQPNP
        )
        
        if not success:
            return None
        
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)
        
        # Get yaw, pitch, roll from rotation matrix
        sy = sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = atan2(rmat[2, 1], rmat[2, 2])
            yaw = atan2(-rmat[2, 0], sy)
            roll = atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = atan2(-rmat[1, 2], rmat[1, 1])
            yaw = atan2(-rmat[2, 0], sy)
            roll = 0
        
        # Convert radians to degrees
        pitch_deg = degrees(pitch)
        yaw_deg = degrees(yaw)
        roll_deg = degrees(roll)
        
        # Compute gaze score based on yaw/pitch
        yaw_score = max(0.0, 1.0 - abs(yaw_deg) / self.max_angle_threshold)
        pitch_score = max(0.0, 1.0 - abs(pitch_deg) / self.max_angle_threshold)
        
        # Combine scores (weighted average)
        final_score = yaw_score*0.9 + pitch_score*0.1
        
        # Create gaze direction vector from rotation matrix
        gaze_direction_3d = -rmat[:, 2]  # Negative Z-axis (forward direction)
        
        return final_score, gaze_direction_3d, pitch_deg, yaw_deg, roll_deg
    
    def update_camera_parameters(self, 
                                focal_length: float,
                                center_x: float,
                                center_y: float):
        """
        Update camera parameters and rebuild camera matrix.
        
        Args:
            focal_length: New focal length
            center_x: New camera center X
            center_y: New camera center Y
        """
        self.focal_length = focal_length
        self.center_x = center_x
        self.center_y = center_y
        
        self.camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype="double")
    
    def set_face_model(self, model_points: List[List[float]]):
        """
        Set custom 3D face model points.
        
        Args:
            model_points: List of 6 3D points [x, y, z] corresponding to:
                         [nose_tip, right_eye, left_eye, right_lip, left_lip, mouth_center]
        """
        self.model_points = np.array(model_points, dtype="double")
    
    def set_method(self, method: str):
        """
        Switch gaze computation method.
        
        Args:
            method: "eye_vector" for stable geometric method or "pnp" for original PnP method
        """
        if method not in ["eye_vector", "pnp"]:
            raise ValueError(f"Unknown method: {method}. Use 'eye_vector' or 'pnp'")
        self.method = method


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two 2D points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Euclidean distance between the points
    """
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalize_landmarks(landmarks: List[Tuple[float, float]], 
                       image_width: int, 
                       image_height: int) -> List[Tuple[float, float]]:
    """
    Convert normalized landmark coordinates to pixel coordinates.
    
    Args:
        landmarks: List of normalized coordinates (0.0 to 1.0)
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        List of pixel coordinates
    """
    pixel_landmarks = []
    for x_norm, y_norm in landmarks:
        pixel_x = x_norm * image_width
        pixel_y = y_norm * image_height
        pixel_landmarks.append((pixel_x, pixel_y))
    
    return pixel_landmarks
