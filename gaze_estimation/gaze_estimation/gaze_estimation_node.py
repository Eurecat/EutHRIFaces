#!/usr/bin/env python3
"""
Gaze Estimation Node for HRI Applications

This node subscribes to FacialLandmarks messages from face detection,
computes gaze direction and score using a pinhole camera model, and
publishes Gaze messages following the ros4hri standard.

Uses the GazeComputer utility class from gaze_utils.py for all gaze computations
to avoid code duplication and improve maintainability.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import numpy as np
import cv2

try:
    from hri_msgs.msg import FacialLandmarks, FacialLandmarksArray, Gaze
except ImportError:
    # Fallback in case hri_msgs is not available
    print("Warning: hri_msgs not found. Please install hri_msgs package.")
    FacialLandmarks = None
    FacialLandmarksArray = None
    Gaze = None
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from typing import Dict, List, Optional, Tuple, Any

from .gaze_utils import GazeComputer

# Math import for gaze visualization
import math


class GazeEstimationNode(Node):
    """
    ROS2 node for gaze estimation from facial landmarks.
    
    Subscribes to FacialLandmarks messages and publishes Gaze messages
    with computed gaze direction and confidence score.
    """
    
    def __init__(self):
        super().__init__('gaze_estimation_node')
        
        # Declare and get parameters
        self.declare_and_get_parameters()
        
        # QoS profile for real-time applications (landmarks and gaze data)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # QoS profile for image data (matching face_detector configuration)
        image_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Create subscriber and publisher
        self.facial_landmarks_sub = self.create_subscription(
            FacialLandmarksArray,
            self.input_topic,
            self.facial_landmarks_array_callback,
            qos_profile
        )
        
        self.gaze_pub = self.create_publisher(
            Gaze,
            self.output_topic,
            qos_profile
        )
        
        # Initialize CV bridge for image handling
        self.bridge = CvBridge()
        self.latest_image = None
        
        # Create image subscriber and publisher if visualization is enabled
        if self.enable_image_output:
            self.image_sub = self.create_subscription(
                Image,
                self.image_input_topic,
                self.image_callback,
                image_qos_profile
            )
            
            self.image_pub = self.create_publisher(
                Image,
                self.image_output_topic,
                10  # Use simple depth like in face_detector
            )
        
        # Initialize gaze computer utility
        self.gaze_computer = GazeComputer(
            focal_length=self.focal_length,
            center_x=self.center_x,
            center_y=self.center_y,
            max_angle_threshold=self.max_angle_threshold
        )
        
        # Setup custom 3D face model from parameters
        self.setup_face_model()
        
        # Rate limiting
        self.last_publish_time = self.get_clock().now()
        self.min_publish_interval = 1.0 / self.publish_rate
        
        self.get_logger().info(f'Gaze Estimation Node started')
        self.get_logger().info(f'Subscribing to: {self.input_topic}')
        self.get_logger().info(f'Publishing to: {self.output_topic}')
        if self.enable_image_output:
            self.get_logger().info(f'Image input topic: {self.image_input_topic}')
            self.get_logger().info(f'Image output topic: {self.image_output_topic}')
        else:
            self.get_logger().info('Image output disabled')
        self.get_logger().info(f'Camera parameters: focal_length={self.focal_length}, '
                              f'center=({self.center_x}, {self.center_y}), '
                              f'image_size=({self.image_width}, {self.image_height})')
    
    def declare_and_get_parameters(self):
        """Declare and get all ROS2 parameters."""
        # Declare and get topic parameters
        self.declare_parameter('input_topic', '/people/faces/detected')
        self.declare_parameter('output_topic', '/people/faces/gaze')
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # Declare and get camera parameters
        # Note: image_width and image_height will be taken from FacialLandmarks message
        self.declare_parameter('focal_length', 640.0)
        self.declare_parameter('center_x_ratio', 0.5)  # Center as ratio of image width
        self.declare_parameter('center_y_ratio', 0.5)  # Center as ratio of image height
        self.focal_length = self.get_parameter('focal_length').get_parameter_value().double_value
        self.center_x_ratio = self.get_parameter('center_x_ratio').get_parameter_value().double_value
        self.center_y_ratio = self.get_parameter('center_y_ratio').get_parameter_value().double_value
        
        # Initialize image dimensions (will be updated from message)
        self.image_width = 640  # Default, will be overwritten
        self.image_height = 480  # Default, will be overwritten
        self.center_x = 320.0  # Default, will be overwritten
        self.center_y = 240.0  # Default, will be overwritten
        
        # Declare and get gaze computation parameters
        self.declare_parameter('max_angle_threshold', 80.0)
        self.declare_parameter('confidence_threshold', 0.1)
        self.max_angle_threshold = self.get_parameter('max_angle_threshold').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        
        # Declare and get 3D face model parameters
        self.declare_parameter('model_points.nose_tip', [0.0, 0.0, 0.0])
        self.declare_parameter('model_points.right_eye', [20.0, -30.0, -20.0])
        self.declare_parameter('model_points.left_eye', [-20.0, -30.0, -20.0])
        self.declare_parameter('model_points.right_lip', [20.0, 30.0, -20.0])
        self.declare_parameter('model_points.left_lip', [-20.0, 30.0, -20.0])
        self.declare_parameter('model_points.mouth_center', [0.0, 30.0, -20.0])
        
        # Declare and get output parameters
        self.declare_parameter('receiver_id', 'pin_hole_cam_model')
        self.receiver_id = self.get_parameter('receiver_id').get_parameter_value().string_value
        
        # Declare and get debug parameters
        self.declare_parameter('enable_debug_output', True)
        self.declare_parameter('publish_rate', 30.0)
        
        # Declare and get image visualization parameters
        self.declare_parameter('enable_image_output', True)
        self.declare_parameter('image_input_topic', '/camera/color/image_rect_raw')
        self.declare_parameter('image_output_topic', '/people/faces/gaze/image_with_gaze')
        self.enable_debug_output = self.get_parameter('enable_debug_output').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.enable_image_output = self.get_parameter('enable_image_output').get_parameter_value().bool_value
        self.image_input_topic = self.get_parameter('image_input_topic').get_parameter_value().string_value
        self.image_output_topic = self.get_parameter('image_output_topic').get_parameter_value().string_value

    
    def setup_camera_matrix(self):
        """Setup the camera intrinsic matrix for the pinhole camera model."""
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center_x],
            [0, self.focal_length, self.center_y],
            [0, 0, 1]
        ], dtype="double")
        
        # No lens distortion assumed
        self.dist_coeffs = np.zeros((4, 1))
    
    def setup_face_model(self):
        """Setup the 3D face model points using GazeComputer utility."""
        # Get 3D model points from parameters
        nose_tip = self.get_parameter('model_points.nose_tip').get_parameter_value().double_array_value
        right_eye = self.get_parameter('model_points.right_eye').get_parameter_value().double_array_value
        left_eye = self.get_parameter('model_points.left_eye').get_parameter_value().double_array_value
        right_lip = self.get_parameter('model_points.right_lip').get_parameter_value().double_array_value
        left_lip = self.get_parameter('model_points.left_lip').get_parameter_value().double_array_value
        mouth_center = self.get_parameter('model_points.mouth_center').get_parameter_value().double_array_value
        
        model_points = [
            nose_tip,      # Nose tip
            right_eye,     # Right eye
            left_eye,      # Left eye  
            right_lip,     # Right lip corner
            left_lip,      # Left lip corner
            mouth_center   # Mouth center
        ]
        
        # Set the model points in the gaze computer
        if hasattr(self, 'gaze_computer'):
            self.gaze_computer.set_face_model(model_points)
    
    def facial_landmarks_array_callback(self, msg):
        """
        Callback for processing array of facial landmarks and computing gaze for all faces.
        
        Args:
            msg: FacialLandmarksArray message containing multiple face landmarks
        """
        if not msg.ids:
            if self.enable_debug_output:
                self.get_logger().debug('Received empty FacialLandmarksArray')
            return
        
        if self.enable_debug_output:
            self.get_logger().debug(f'Processing FacialLandmarksArray with {len(msg.ids)} faces')
        
        # Process each face in the array
        for facial_landmarks_msg in msg.ids:
            try:
                self.process_single_face_landmarks(facial_landmarks_msg)
            except Exception as e:
                self.get_logger().error(f'Error processing face {facial_landmarks_msg.face_id}: {str(e)}')
    
    def process_single_face_landmarks(self, msg):
        """
        Process a single facial landmarks message (extracted from the original callback).
        
        Args:
            msg: FacialLandmarks message containing face landmarks
        """
        # Update image dimensions and camera parameters from message
        self.update_camera_parameters_from_message(msg)
        
        # Rate limiting per face (optional - you might want to remove this for batch processing)
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_publish_time).nanoseconds / 1e9
        if time_diff < self.min_publish_interval:
            return
        
        try:
            # Extract gaze information
            gaze_result = self.compute_gaze_from_landmarks(msg)
            
            if gaze_result is not None:
                gaze_score, gaze_direction, pitch, yaw, roll = gaze_result
                
                # Create and publish Gaze message
                gaze_msg = Gaze()
                gaze_msg.header = Header()
                gaze_msg.header.stamp = self.get_clock().now().to_msg()
                gaze_msg.header.frame_id = msg.header.frame_id
                
                gaze_msg.sender = msg.face_id
                gaze_msg.receiver = self.receiver_id
                gaze_msg.score = float(gaze_score)
                gaze_msg.gaze_direction = gaze_direction
                
                self.gaze_pub.publish(gaze_msg)
                self.last_publish_time = current_time
                
                # Publish gaze visualization if enabled
                if self.enable_image_output:
                    self.publish_gaze_visualization(msg, gaze_score, gaze_direction, pitch, yaw, roll)
                
                if self.enable_debug_output:
                    self.get_logger().debug(
                        f'Face {msg.face_id}: gaze_score={gaze_score:.3f}, '
                        f'yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°'
                    )
            
        except Exception as e:
            self.get_logger().error(f'Error processing facial landmarks: {str(e)}')

    def image_callback(self, msg: Image):
        """
        Callback for receiving RGB images for gaze visualization.
        
        Args:
            msg: sensor_msgs/Image message
        """
        try:
            # Store the latest image for gaze visualization
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.enable_debug_output:
                self.get_logger().debug(f'Received image: {self.latest_image.shape if self.latest_image is not None else "None"}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def publish_gaze_visualization(self, landmarks_msg, gaze_score: float, 
                                 gaze_direction, pitch: float, yaw: float, roll: float):
        """
        Publish image with gaze visualization overlay.
        
        Args:
            landmarks_msg: FacialLandmarks message
            gaze_score: Computed gaze confidence score
            gaze_direction: Gaze direction vector
            pitch: Head pitch angle in degrees
            yaw: Head yaw angle in degrees  
            roll: Head roll angle in degrees
        """
        if not self.enable_image_output or self.latest_image is None:
            if self.enable_debug_output:
                self.get_logger().debug(f'Skipping gaze visualization: enable_image_output={self.enable_image_output}, latest_image={self.latest_image is not None}')
            return
            
        try:
            # Create a copy for annotation
            annotated_image = self.latest_image.copy()
            
            # Extract face bounding box from landmarks message
            if len(landmarks_msg.bbox_xyxy) >= 4:
                x1, y1, x2, y2 = landmarks_msg.bbox_xyxy[:4]
                face_bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h] format
                
                # Draw gaze visualization
                self._draw_gaze_on_image(annotated_image, face_bbox, landmarks_msg,
                                       gaze_score, gaze_direction, pitch, yaw, roll)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f'Drew gaze visualization for face {landmarks_msg.face_id}')
            
            # Convert back to ROS Image and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = landmarks_msg.header
            self.image_pub.publish(annotated_msg)
            
            if self.enable_debug_output:
                self.get_logger().debug(f'Published gaze visualization image')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing gaze visualization: {str(e)}')
    
    def _draw_gaze_on_image(self, image: np.ndarray, face_bbox: List[int], 
                          landmarks_msg, gaze_score: float, gaze_direction,
                          pitch: float, yaw: float, roll: float):
        """
        Draw gaze direction and information on the image.
        
        Args:
            image: OpenCV image to draw on
            face_bbox: Face bounding box [x, y, w, h]
            landmarks_msg: FacialLandmarks message
            gaze_score: Gaze confidence score [0.0, 1.0]
            gaze_direction: 3D gaze direction vector
            pitch, yaw, roll: Head pose angles in degrees
        """
        try:
            # Calculate adaptive sizes based on image dimensions
            img_height, img_width = image.shape[:2]
            base_size = min(img_width, img_height)
            current_scale_factor = base_size / 640.0  # Scale based on 640px reference
            
            # Get face bbox coordinates
            x, y, w, h = face_bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Choose color based on gaze score quality
            if gaze_score > 0.7:
                color = (0, 255, 0)  # Green for good gaze (looking at camera)
            elif gaze_score > 0.4:
                color = (0, 255, 255)  # Yellow for moderate gaze
            else:
                color = (0, 100, 255)  # Orange for poor gaze (looking away)
            
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 * current_scale_factor
            thickness = max(1, int(2 * current_scale_factor))
            
            # Draw face bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Format gaze score text
            gaze_text = f"Face: {landmarks_msg.face_id} | Gaze: {gaze_score:.2f}"
            
            # Position text above the face bbox
            text_x = x1
            text_y = y1 - 10
            
            # Get text size for background rectangle
            text_size = cv2.getTextSize(gaze_text, font, font_scale, thickness)[0]
            
            # Draw background rectangle for better readability
            padding = max(1, int(3 * current_scale_factor))
            cv2.rectangle(image, 
                        (text_x - padding, text_y - text_size[1] - padding), 
                        (text_x + text_size[0] + padding, text_y + padding), 
                        (0, 0, 0), -1)  # Black background
            
            # Draw gaze score text
            cv2.putText(image, gaze_text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Draw pose information below the face bbox
            pose_text = f"Y:{yaw:.0f}° P:{pitch:.0f}° R:{roll:.0f}°"
            pose_y = y2 + 20
            cv2.putText(image, pose_text, (x1, pose_y), font, font_scale * 0.8, color, thickness)
            
            # Draw a small indicator circle at face center
            circle_radius = max(1, int(3 * current_scale_factor))
            cv2.circle(image, (center_x, center_y), circle_radius, color, -1)
            
            # Draw head pose axes
            self._draw_head_pose_axes(image, center_x, center_y, pitch, yaw, roll, 
                                    w, h, current_scale_factor)
            
            # Draw gaze direction arrow
            gaze_length = max(40, int(min(w, h) * 0.8))  # Scale with face size
            arrow_thickness = max(2, int(3 * current_scale_factor))
            
            # Calculate gaze endpoint using 2D projection of 3D direction
            end_x = int(center_x + gaze_direction.x * gaze_length)
            end_y = int(center_y + gaze_direction.y * gaze_length)
            
            # Draw gaze arrow in bright cyan for visibility
            cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), 
                           (255, 255, 0), arrow_thickness, tipLength=0.3)
            
            # Draw facial landmarks
            self._draw_facial_landmarks(image, landmarks_msg, current_scale_factor)
            
        except Exception as e:
            self.get_logger().error(f'Error drawing gaze visualization: {str(e)}')

    def _draw_head_pose_axes(self, image: np.ndarray, center_x: int, center_y: int,
                           pitch: float, yaw: float, roll: float, face_w: int, face_h: int,
                           current_scale_factor: float):
        """
        Draw head pose axes (X=red, Y=green, Z=blue) at the face center.
        
        Args:
            image: OpenCV image to draw on
            center_x, center_y: Face center coordinates
            pitch, yaw, roll: Head pose angles in degrees
            face_w, face_h: Face bounding box dimensions
            current_scale_factor: Scale factor for drawing
        """
        try:
            # Length of each axis line in pixels
            axis_length = int(min(face_w, face_h) * 0.5)
            
            # Convert angles from degrees to radians
            pitch_rad = math.radians(pitch)
            yaw_rad = math.radians(yaw)
            roll_rad = math.radians(roll)
            
            # Precompute sin/cos
            sin_pitch = math.sin(pitch_rad)
            cos_pitch = math.cos(pitch_rad)
            sin_yaw = math.sin(yaw_rad)
            cos_yaw = math.cos(yaw_rad)
            sin_roll = math.sin(roll_rad)
            cos_roll = math.cos(roll_rad)
            
            # Calculate rotated axis vectors (projected to 2D)
            # X axis (red): pointing right
            x_axis_x = axis_length * (cos_yaw * cos_roll)
            x_axis_y = axis_length * (cos_pitch * sin_roll + sin_pitch * sin_yaw * cos_roll)
            
            # Y axis (green): pointing down
            y_axis_x = axis_length * (-cos_yaw * sin_roll)
            y_axis_y = axis_length * (cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll)
            
            # Z axis (blue): pointing out of face (forward)
            z_axis_x = axis_length * (sin_yaw)
            z_axis_y = axis_length * (-sin_pitch * cos_yaw)
            
            # Convert to int for drawing
            pt_center = (center_x, center_y)
            pt_x = (int(center_x + x_axis_x), int(center_y + x_axis_y))
            pt_y = (int(center_x + y_axis_x), int(center_y + y_axis_y))
            pt_z = (int(center_x + z_axis_x), int(center_y + z_axis_y))
            
            # Draw the axes arrows
            arrow_thickness = max(1, int(2 * current_scale_factor))
            arrow_tip_length = 0.2
            
            # X axis in red
            cv2.arrowedLine(image, pt_center, pt_x, (0, 0, 255), arrow_thickness, tipLength=arrow_tip_length)
            # Y axis in green
            cv2.arrowedLine(image, pt_center, pt_y, (0, 255, 0), arrow_thickness, tipLength=arrow_tip_length)
            # Z axis in blue
            cv2.arrowedLine(image, pt_center, pt_z, (255, 0, 0), arrow_thickness, tipLength=arrow_tip_length)
            
        except Exception as e:
            self.get_logger().error(f'Error drawing head pose axes: {str(e)}')

    def _draw_facial_landmarks(self, image: np.ndarray, landmarks_msg, current_scale_factor: float):
        """
        Draw facial keypoint landmarks on the image (eyes, nose, mouth, face outline).
        Args:
            image: OpenCV image to draw on
            landmarks_msg: FacialLandmarks message containing landmarks
            current_scale_factor: Scale factor for drawing
        """
        try:
            if not hasattr(landmarks_msg, 'landmarks') or not landmarks_msg.landmarks:
                return
            circle_radius = max(1, int(2 * current_scale_factor))
            # Define landmark groups based on ros4hri FacialLandmarks indices
            eye_landmarks = list(range(36, 48))  # Eyes (36-47)
            nose_landmarks = list(range(27, 36))  # Nose (27-35)
            mouth_landmarks = list(range(48, 68))  # Mouth (48-67)
            face_outline_landmarks = list(range(0, 17)) + list(range(17, 27))  # Face contour (0-26)
            # Colors
            colors = {
                'eyes': (255, 255, 0),      # Cyan
                'nose': (0, 255, 255),      # Yellow
                'mouth': (255, 0, 255),     # Magenta
                'face': (128, 128, 128),    # Gray
                'default': (255, 255, 255)  # White
            }
            for i, landmark in enumerate(landmarks_msg.landmarks):
                if getattr(landmark, 'c', 0.0) > 0.0:
                    pixel_x = int(landmark.x * landmarks_msg.width)
                    pixel_y = int(landmark.y * landmarks_msg.height)
                    if i in eye_landmarks:
                        color = colors['eyes']
                    elif i in nose_landmarks:
                        color = colors['nose']
                    elif i in mouth_landmarks:
                        color = colors['mouth']
                    elif i in face_outline_landmarks:
                        color = colors['face']
                    else:
                        color = colors['default']
                    cv2.circle(image, (pixel_x, pixel_y), circle_radius, color, -1)
                    # Optionally draw index for key points
                    if self.enable_debug_output and i in [30, 36, 39, 42, 45, 48, 54]:
                        font_scale = 0.3 * current_scale_factor
                        cv2.putText(image, str(i), (pixel_x + 5, pixel_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        except Exception as e:
            self.get_logger().error(f'Error drawing facial landmarks: {str(e)}')
    
    def compute_gaze_from_landmarks(self, landmarks_msg) -> Optional[Tuple[float, Any, float, float, float]]:
        """
        Compute gaze score and direction from facial landmarks using GazeComputer utility.
        
        Args:
            landmarks_msg: FacialLandmarks message
            
        Returns:
            Tuple of (gaze_score, gaze_direction_vector, pitch, yaw, roll) or None
        """
        if not landmarks_msg.landmarks:
            if self.enable_debug_output:
                self.get_logger().warn('No landmarks in message')
            return None
        
        # Extract required landmarks from the ros4hri FacialLandmarks message
        landmark_points = self.extract_key_landmarks(landmarks_msg)
        
        if landmark_points is None:
            return None
        
        nose, right_eye, left_eye, right_lip, left_lip = landmark_points
        
        # Use the GazeComputer utility to compute gaze
        gaze_result = self.gaze_computer.compute_gaze(
            nose=nose, 
            right_eye=right_eye, 
            left_eye=left_eye,
            right_lip=right_lip, 
            left_lip=left_lip
        )
        
        if gaze_result is None:
            if self.enable_debug_output:
                self.get_logger().warn('Gaze computation failed')
            return None
        
        gaze_score, gaze_direction_3d, pitch_deg, yaw_deg, roll_deg = gaze_result
        
        # Convert numpy array to ROS Vector3 message
        gaze_direction = Vector3()
        gaze_direction.x = float(gaze_direction_3d[0])
        gaze_direction.y = float(gaze_direction_3d[1])
        gaze_direction.z = float(gaze_direction_3d[2])
        
        return gaze_score, gaze_direction, pitch_deg, yaw_deg, roll_deg
    
    def extract_key_landmarks(self, landmarks_msg) -> Optional[Tuple]:
        """
        Extract the 5 key landmarks needed for gaze estimation from ros4hri FacialLandmarks.
        
        Based on the landmark codes defined in FacialLandmarks.msg:
        - LEFT_EYE_INSIDE = 42 (or use LEFT_PUPIL = 69 if available)
        - RIGHT_EYE_INSIDE = 39 (or use RIGHT_PUPIL = 68 if available)  
        - NOSE = 30
        - MOUTH_OUTER_LEFT = 54
        - MOUTH_OUTER_RIGHT = 48
        
        Args:
            landmarks_msg: FacialLandmarks message
            
        Returns:
            Tuple of (nose, right_eye, left_eye, right_lip, left_lip) as pixel coordinates
            or None if required landmarks are not available
        """
        # Create a dictionary for easy landmark lookup
        landmark_dict = {}
        for i, landmark in enumerate(landmarks_msg.landmarks):
            # Only include landmarks with confidence > 0
            if landmark.c > 0.0:
                # Convert normalized coordinates to pixel coordinates
                pixel_x = landmark.x * landmarks_msg.width
                pixel_y = landmark.y * landmarks_msg.height
                landmark_dict[i] = (pixel_x, pixel_y)
        
        # Define the landmark indices we need (from FacialLandmarks.msg constants)
        try:
            # Try to get pupil positions first (more accurate for gaze)
            if 68 in landmark_dict and 69 in landmark_dict:
                right_eye = landmark_dict[68]  # RIGHT_PUPIL
                left_eye = landmark_dict[69]   # LEFT_PUPIL
            else:
                # Fall back to eye inner corners
                right_eye = landmark_dict[39]  # RIGHT_EYE_INSIDE
                left_eye = landmark_dict[42]   # LEFT_EYE_INSIDE
            
            nose = landmark_dict[30]       # NOSE
            right_lip = landmark_dict[48]  # MOUTH_OUTER_RIGHT
            left_lip = landmark_dict[54]   # MOUTH_OUTER_LEFT
            
            return nose, right_eye, left_eye, right_lip, left_lip
            
        except KeyError as e:
            missing_landmark = int(str(e).strip("'"))
            if self.enable_debug_output:
                self.get_logger().warn(f'Missing required landmark {missing_landmark}')
            return None
    
    def update_camera_parameters_from_message(self, landmarks_msg):
        """
        Update camera parameters based on image dimensions from FacialLandmarks message.
        
        Args:
            landmarks_msg: FacialLandmarks message containing image dimensions
        """
        # Update image dimensions from message
        if landmarks_msg.width != self.image_width or landmarks_msg.height != self.image_height:
            self.image_width = landmarks_msg.width
            self.image_height = landmarks_msg.height
            
            # Update camera center based on ratios
            self.center_x = self.center_x_ratio * self.image_width
            self.center_y = self.center_y_ratio * self.image_height
            
            # Update focal length if it should match image width
            if abs(self.focal_length - 640.0) < 1.0:  # Default focal length
                self.focal_length = float(self.image_width)
            
            # Update gaze computer with new camera parameters
            self.gaze_computer.update_camera_parameters(
                focal_length=self.focal_length,
                center_x=self.center_x,
                center_y=self.center_y
            )
            
            if self.enable_debug_output:
                self.get_logger().info(f'Updated camera parameters: '
                                     f'image=({self.image_width}x{self.image_height}), '
                                     f'center=({self.center_x:.1f}, {self.center_y:.1f}), '
                                     f'focal_length={self.focal_length:.1f}')

def main(args=None):
    """Main function for the gaze estimation node."""
    rclpy.init(args=args)
    
    try:
        node = GazeEstimationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in gaze estimation node: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
