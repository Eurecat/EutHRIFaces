#!/usr/bin/env python3
"""
Gaze Estimation Node for HRI Applications

This node subscribes to FacialLandmarksArray messages from face detection,
computes gaze direction and score using a pinhole camera model, and
publishes Gaze messages following the ros4hri standard.

Uses the GazeComputer utility class from gaze_utils.py for all gaze computations
to avoid code duplication and improve maintainability.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time

import numpy as np
import cv2
import time

try:
    from hri_msgs.msg import FacialLandmarks, FacialLandmarksArray, Gaze, GazeArray, IdsList
except ImportError:
    # Fallback in case hri_msgs is not available
    print("Warning: hri_msgs not found. Please install hri_msgs package.")
    FacialLandmarks = None
    FacialLandmarksArray = None
    Gaze = None
    GazeArray = None
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from typing import Dict, List, Optional, Tuple, Any

from .gaze_utils import GazeComputer

# Math import for gaze visualization
import math


class GazeEstimationNode(Node):
    """
    ROS2 node for gaze estimation from facial landmarks.
    
    Subscribes to FacialLandmarksArray messages and publishes Gaze messages
    with computed gaze direction and confidence score.
    """
    
    def __init__(self):
        super().__init__('gaze_estimation_node')
        
        # Timing statistics
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.max_processing_time = 0.0
        self.min_processing_time = float('inf')
        
        # Declare and get parameters
        self.declare_and_get_parameters()

        # Initialize image storage variables (copied from perception node)
        self.latest_color_image_msg = None
        self.color_image_processed = False
        self.latest_color_image_timestamp = None
        
        # Setup QoS profiles (copied from perception node)
        self.qos_profile = QoSProfile(
            depth=1,  # Keep only the latest image
            # reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # durability=DurabilityPolicy.VOLATILE,
            # # history=QoSHistoryPolicy.KEEP_LAST,
        )
        # Create subscriber and publisher based on mode
        if self.ros4hri_with_id:
            # ROS4HRI with ID mode: Subscribe to tracked faces list and per-ID topics
            # Dictionary to store subscribers and publishers for each face ID
            self.landmarks_subscribers = {}  # {face_id: Subscription}
            self.gaze_publishers = {}  # {face_id: Publisher}
            self.tracked_face_ids = set()  # Set of currently tracked face IDs
            
            # Subscribe to tracked faces list
            self.tracked_faces_sub = self.create_subscription(
                IdsList,
                '/humans/faces/tracked',
                self.tracked_faces_callback,
                self.qos_profile
            )
            self.get_logger().info("ROS4HRI with ID mode enabled: Subscribing to /humans/faces/tracked and per-ID topics")
            self.facial_landmarks_sub = None
        else:
            # ROS4HRI array mode: Subscribe to FacialLandmarksArray messages
            self.facial_landmarks_sub = self.create_subscription(
                FacialLandmarksArray,
                self.input_topic,
                self.facial_landmarks_array_callback,
                self.qos_profile
            )
            self.get_logger().info("ROS4HRI array mode enabled: Subscribing to FacialLandmarksArray messages")
            self.landmarks_subscribers = {}
            self.gaze_publishers = {}
            self.tracked_face_ids = set()
            self.tracked_faces_sub = None
        
        # Create publisher based on mode - only one publisher per topic
        if self.ros4hri_with_id:
            # ROS4HRI with ID mode: Publishers will be created dynamically per face ID
            # (gaze_publishers dictionary is already initialized above)
            self.gaze_pub = None
        else:
            # ROS4HRI array mode: Publish GazeArray
            self.gaze_pub = self.create_publisher(
                GazeArray,
                self.output_topic,
                self.qos_profile
            )
        
        if self.ros4hri_with_id:
            self.get_logger().info("Publishing individual Gaze messages")
        else:
            self.get_logger().info("Publishing GazeArray messages")
        
        # Initialize CV bridge for image handling
        self.bridge = CvBridge()
        
        # Create RGB subscriber - choose between compressed and regular image
        if self.enable_image_output:
            self.get_logger().info("Setting up RGB processing for gaze visualization")
            if self.compressed_topic and self.compressed_topic.strip():
                self.get_logger().info(f"Using compressed image topic: {self.compressed_topic}")
                real_time_qos = QoSProfile(
                    depth=1,  # Keep only latest image
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # No retransmissions
                    history=QoSHistoryPolicy.KEEP_LAST,
                    durability=DurabilityPolicy.VOLATILE  # Don't persist messages
                )
                self.color_sub = self.create_subscription(
                    CompressedImage, 
                    self.compressed_topic, 
                    self._store_latest_compressed_rgb, 
                    real_time_qos
                )
            else:
                self.get_logger().info(f"Using regular image topic: {self.image_input_topic}")
                self.color_sub = self.create_subscription(
                    Image, 
                    self.image_input_topic, 
                    self._store_latest_rgb, 
                    self.qos_profile
                )
            
            image_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,   # 1–5 is ideal for images over Wi-Fi
                durability=DurabilityPolicy.VOLATILE
            )
            self.image_pub = self.create_publisher(
                Image,
                self.output_image_topic,
                image_qos
            )

        # Timer for periodic inference (copied from perception node pattern)
        timer_period = 1.0 / self.processing_rate_hz  # Use processing_rate_hz parameter
        self.inference_timer = self.create_timer(
            timer_period, 
            self.inference_timer_callback
        )
        
        # Timer for processing buffered frames in ROS4HRI with ID mode
        if self.ros4hri_with_id:
            buffer_timer_period = 0.05  # Check buffer every 50ms
            self.buffer_timer = self.create_timer(
                buffer_timer_period,
                self.process_buffered_frames
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

        # Store latest landmarks for processing (array mode)
        self.latest_landmarks_array = None
        self.landmarks_processed = False
        
        # Message buffer for ROS4HRI with ID mode (sync by timestamp)
        self.landmarks_buffer = {}  # {timestamp: [FacialLandmarks, ...]}
        self.buffer_timeout = 0.1  # 100ms timeout for frame synchronization
        
        self.get_logger().info(f'\033[92m[INFO] Gaze estimation model initialized successfully\033[0m')
        self.get_logger().info(f'Gaze Estimation Node started')
        self.get_logger().info(f'Processing rate: {self.processing_rate_hz} Hz')
        self.get_logger().info(f'Subscribing to: {self.input_topic}')
        self.get_logger().info(f'Publishing to: {self.output_topic}')
        if self.enable_image_output:
            self.get_logger().info(f'Image input topic: {self.image_input_topic}')
            self.get_logger().info(f'Image output topic: {self.output_image_topic}')
        else:
            self.get_logger().info('Image output disabled')
        self.get_logger().info(f'Camera parameters: focal_length={self.focal_length}, '
                              f'center=({self.center_x}, {self.center_y}), '
                              f'image_size=({self.image_width}, {self.image_height})')

    # -------------------------------------------------------------------------
    #             Image Storage Callbacks (Copied from perception node)
    # -------------------------------------------------------------------------
    def _store_latest_rgb(self, color_msg):
        """
        Stores the latest color image.

        Simple callback for RGB image storage.

        Args:
            color_msg: ROS Image message containing RGB data
        """
        self.latest_color_image_msg = color_msg
        self.color_image_processed = False
        self.latest_color_image_timestamp = self.get_clock().now()
        # self.get_logger().info("Color image received.")

    def _store_latest_compressed_rgb(self, color_msg):
        """
        Stores the latest compressed color image.

        Simple callback for compressed RGB image storage.

        Args:
            color_msg: ROS CompressedImage message containing RGB data
        """
        self.latest_color_image_msg = color_msg
        self.color_image_processed = False
        self.latest_color_image_timestamp = self.get_clock().now()
        # self.get_logger().info("Compressed color image received.")

    # -------------------------------------------------------------------------
    #                         Timer Callback for Inference
    # -------------------------------------------------------------------------
    def inference_timer_callback(self):
        """
        Regular callback for continuous inference mode.

        Triggered by the timer at the configured frequency.
        Processes latest landmarks and image data for gaze estimation.
        """
        
        start_time = self.get_clock().now()

        # Check if we have new landmarks to process
        if self.latest_landmarks_array is None:
            # self.get_logger().warning("No landmarks data received")
            return
        if self.landmarks_processed is True:
            return

        # If image visualization is enabled, check for image data
        if self.enable_image_output:
            color_msg = self.latest_color_image_msg
            color_image_processed = self.color_image_processed
            
            if color_msg is None:
                # self.get_logger().warning("No image data received for visualization")
                return
            if color_image_processed is True:
                return
            
            # Convert image to OpenCV format
            try:
                if self.compressed_topic and self.compressed_topic.strip():
                    # Handle compressed image
                    np_arr = np.frombuffer(color_msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if cv_image is None:
                        self.get_logger().error('Failed to decode compressed image')
                        return
                else:
                    # Handle regular image  
                    cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f'Error converting image: {e}')
                return
            
            self.color_image_processed = True
            
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("Received empty or invalid image")
                return
        else:
            cv_image = None

        # Mark landmarks as processed
        landmarks_msg = self.latest_landmarks_array
        self.landmarks_processed = True

        try:
            # Process the landmarks array
            self.process_landmarks_array(landmarks_msg, cv_image)
                
        except Exception as e:
            self.get_logger().error(f"Error processing gaze estimation: {e}")

        # Calculate and log timing information
        end_time = self.get_clock().now()
        processing_time = (end_time - start_time).nanoseconds / 1e6  # Convert to milliseconds
        
        # Update timing statistics
        self.frame_count += 1
        self.total_processing_time += processing_time
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.min_processing_time = min(self.min_processing_time, processing_time)
        
        # Log timing every 200 frames or when debug is enabled
        if self.frame_count % 200 == 0 or self.enable_debug_output:
            avg_time = self.total_processing_time / self.frame_count
            faces_count = len(landmarks_msg.ids) if landmarks_msg and landmarks_msg.ids else 0
            self.get_logger().info(
                f"[TIMING] Gaze Estimation - Frame #{self.frame_count}: "
                f"Current: {processing_time:.2f}ms, "
                f"Avg: {avg_time:.2f}ms, "
                f"Min: {self.min_processing_time:.2f}ms, "
                f"Max: {self.max_processing_time:.2f}ms, "
                f"Faces: {faces_count}"
            )

    def process_landmarks_array(self, landmarks_msg, cv_image=None):
        """
        Process landmarks array and compute gaze for all faces.
        
        Args:
            landmarks_msg: FacialLandmarksArray message
            cv_image: OpenCV image for visualization (optional)
        """
        if not landmarks_msg.ids:
            if self.enable_debug_output:
                self.get_logger().debug('Received empty FacialLandmarksArray')
            return

        if self.enable_debug_output:
            self.get_logger().debug(f'Processing FacialLandmarksArray with {len(landmarks_msg.ids)} faces')
        
        # Update camera parameters once per frame using the first face's message
        # This assumes all faces in the same frame have the same image dimensions
        if landmarks_msg.ids:
            self.update_camera_parameters_from_message(landmarks_msg.ids[0])
        
        gaze_array_msg = GazeArray()
        gaze_array_msg.header = landmarks_msg.header if landmarks_msg.header is not None else Header()

        # For visualization - we'll collect gaze data for all faces
        gaze_visualization_data = []
        
        try:
            # Process each face in the array
            for facial_landmarks_msg in landmarks_msg.ids:
                try:
                    gaze_msg, gaze_data = self.process_single_face_landmarks(facial_landmarks_msg)
                    if gaze_msg:
                        gaze_array_msg.gaze_array.append(gaze_msg)
                        if gaze_data:
                            gaze_visualization_data.append((facial_landmarks_msg, gaze_data))
                except Exception as e:
                    self.get_logger().error(f'Error processing face {facial_landmarks_msg.face_id}: {str(e)}')
            
            # Publish gaze results based on mode
            if self.ros4hri_with_id:
                # ROS4HRI with ID mode: Publish to per-ID topics /humans/faces/<faceID>/gaze
                # All messages from the same frame share the same timestamp for synchronization
                frame_timestamp = landmarks_msg.header.stamp if landmarks_msg.header is not None else None
                for gaze_msg, gaze_data in zip(gaze_array_msg.gaze_array, gaze_visualization_data):
                    face_id = gaze_msg.sender
                    
                    # Create publisher for this face ID if it doesn't exist
                    if face_id not in self.gaze_publishers:
                        topic_name = f'/humans/faces/{face_id}/gaze'
                        self.gaze_publishers[face_id] = self.create_publisher(
                            Gaze,
                            topic_name,
                            self.qos_profile
                        )
                        if self.enable_debug_output:
                            self.get_logger().debug(f"Created publisher for face ID: {topic_name}")
                    
                    # Ensure all messages from the same frame have the same timestamp
                    if frame_timestamp:
                        gaze_msg.header.stamp = frame_timestamp
                    
                    # Publish to the per-ID topic
                    self.gaze_publishers[face_id].publish(gaze_msg)
                    if self.enable_debug_output:
                        self.get_logger().debug(f"Published Gaze for face_id={face_id} to /humans/faces/{face_id}/gaze with timestamp={frame_timestamp}")
            else:
                # ROS4HRI array mode: Publish GazeArray message
                self.gaze_pub.publish(gaze_array_msg)
                if self.enable_debug_output:
                    self.get_logger().debug(f"Published GazeArray with {len(gaze_array_msg.gaze_array)} faces")
            
            # Handle image visualization outside the per-face processing
            if self.enable_image_output and cv_image is not None and gaze_visualization_data:
                annotated_image = cv_image.copy()
                for face_msg, (gaze_score, gaze_direction, pitch, yaw, roll) in gaze_visualization_data:
                    self.draw_single_face_visualization(annotated_image, face_msg, gaze_score, gaze_direction, pitch, yaw, roll)
                
                # Convert to ROS Image and publish once
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = landmarks_msg.header if landmarks_msg.header is not None else Header()
                self.image_pub.publish(annotated_msg)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f'Published gaze visualization with {len(gaze_visualization_data)} faces')
        
        except Exception as e:
            self.get_logger().error(f'Error in facial landmarks array processing: {e}')
    
    def declare_and_get_parameters(self):
        """Declare and get all ROS2 parameters."""
        # Declare and get topic parameters
        self.declare_parameter('compressed_topic', '')
        self.declare_parameter('input_topic', '/humans/faces/detected')
        self.declare_parameter('output_topic', '/humans/faces/gaze')
        self.compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # Processing rate parameter (copied from perception node)
        self.declare_parameter('processing_rate_hz', 30.0)  # Default 30 Hz
        self.processing_rate_hz = self.get_parameter('processing_rate_hz').get_parameter_value().double_value
        
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
        self.declare_parameter('output_image_topic', '/humans/faces/gaze/annotated_img')
        
        # ROS4HRI mode parameter - when enabled, subscribes to per-ID messages and publishes per-ID
        self.declare_parameter('ros4hri_with_id', False)  # Default to array mode (ROS4HRI array)
        
        self.enable_debug_output = self.get_parameter('enable_debug_output').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.enable_image_output = self.get_parameter('enable_image_output').get_parameter_value().bool_value
        self.image_input_topic = self.get_parameter('image_input_topic').get_parameter_value().string_value
        self.output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        self.ros4hri_with_id = self.get_parameter('ros4hri_with_id').get_parameter_value().bool_value

    
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
        # Store the latest landmarks array for processing
        self.latest_landmarks_array = msg
        self.landmarks_processed = False
    
    def tracked_faces_callback(self, msg):
        """
        Callback for tracked faces list in ROS4HRI with ID mode.
        Manages dynamic subscriptions to per-ID topics.
        
        Args:
            msg: IdsList message containing currently tracked face IDs
        """
        new_tracked_ids = set(msg.ids)
        
        # Add new face IDs - create subscribers and publishers
        for face_id in new_tracked_ids:
            if face_id not in self.tracked_face_ids:
                # Create subscriber for this face ID
                topic_name = f'/humans/faces/{face_id}/detected'
                self.landmarks_subscribers[face_id] = self.create_subscription(
                    FacialLandmarks,
                    topic_name,
                    lambda m, fid=face_id: self.facial_landmarks_individual_callback(m, fid),
                    self.qos_profile
                )
                
                # Create publisher for this face ID
                output_topic_name = f'/humans/faces/{face_id}/gaze'
                self.gaze_publishers[face_id] = self.create_publisher(
                    Gaze,
                    output_topic_name,
                    self.qos_profile
                )
                
                self.get_logger().debug(f"Subscribed to {topic_name} and publishing to {output_topic_name}")
        
        # Remove old face IDs (cleanup - ROS2 doesn't allow destroying subscriptions, but we can track them)
        removed_ids = self.tracked_face_ids - new_tracked_ids
        for face_id in removed_ids:
            if self.enable_debug_output:
                self.get_logger().debug(f"Face ID {face_id} no longer tracked")
        
        # Update tracked face IDs
        self.tracked_face_ids = new_tracked_ids
    
    def facial_landmarks_individual_callback(self, msg, face_id):
        """
        Callback for processing individual FacialLandmarks messages from per-ID topics in ROS4HRI with ID mode.
        Messages are buffered by timestamp and processed in batch when a complete frame is received.
        
        Args:
            msg: FacialLandmarks message for a single face
            face_id: Face ID (for compatibility with lambda)
        """
        # Get timestamp as a key for frame synchronization
        timestamp_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        
        # Add message to buffer
        if timestamp_key not in self.landmarks_buffer:
            self.landmarks_buffer[timestamp_key] = []
        self.landmarks_buffer[timestamp_key].append(msg)
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Buffered FacialLandmarks for face_id={msg.face_id}, timestamp={timestamp_key}, buffer_size={len(self.landmarks_buffer[timestamp_key])}")
    
    def process_buffered_frames(self):
        """
        Process buffered frames that are ready (older than buffer_timeout or complete frame detected).
        This is called periodically by a timer in ROS4HRI with ID mode.
        """
        if not self.ros4hri_with_id or not self.landmarks_buffer:
            return
        
        current_time = self.get_clock().now()
        current_ns = current_time.nanoseconds
        frames_to_process = []
        
        # Find frames that are ready to process (older than buffer_timeout)
        for ts_key, messages in list(self.landmarks_buffer.items()):
            # Calculate time difference manually to avoid clock type issues
            ts_ns = int(ts_key[0]) * int(1e9) + int(ts_key[1])
            ts_diff = (current_ns - ts_ns) / 1e9  # Convert to seconds
            
            # Process if frame is old enough (all messages from same frame should have arrived)
            if ts_diff > self.buffer_timeout:
                frames_to_process.append((ts_key, messages))
        
        # Process ready frames
        for ts_key, messages in frames_to_process:
            # Create a virtual FacialLandmarksArray from buffered messages
            landmarks_array_msg = FacialLandmarksArray()
            landmarks_array_msg.header = messages[0].header  # Use header from first message
            landmarks_array_msg.ids = messages
            
            # Process as if it were an array message
            self.latest_landmarks_array = landmarks_array_msg
            self.landmarks_processed = False
            
            # Remove from buffer
            del self.landmarks_buffer[ts_key]
            
            if self.enable_debug_output:
                self.get_logger().debug(f"Processing buffered frame with {len(messages)} faces (timestamp: {ts_key})")

    def process_single_face_landmarks(self, msg):
        """
        Process a single facial landmarks message.
        
        Args:
            msg: FacialLandmarks message containing face landmarks
            
        Returns:
            Tuple of (gaze_msg, gaze_data) where:
                - gaze_msg is the ROS Gaze message or None
                - gaze_data is a tuple of (gaze_score, gaze_direction, pitch, yaw, roll) or None
        """
        try:
            # Extract gaze information
            gaze_result = self.compute_gaze_from_landmarks(msg)
            
            if gaze_result is not None:
                gaze_score, gaze_direction, pitch, yaw, roll = gaze_result
                
                # Create Gaze message
                gaze_msg = Gaze()
                gaze_msg.header = msg.header if msg.header is not None else Header()
                
                gaze_msg.sender = msg.face_id
                gaze_msg.receiver = self.receiver_id
                gaze_msg.score = float(gaze_score)
                gaze_msg.gaze_direction = gaze_direction
                
                if self.enable_debug_output:
                    self.get_logger().debug(
                        f'Face {msg.face_id}: gaze_score={gaze_score:.3f}, '
                        f'yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°'
                    )
                
                # Return both the message and the gaze data for visualization
                return gaze_msg, (gaze_score, gaze_direction, pitch, yaw, roll)
            
            return None, None
            
        except Exception as e:
            self.get_logger().error(f'Error processing facial landmarks: {str(e)}')
            return None, None

    def draw_single_face_visualization(self, image, landmarks_msg, gaze_score: float, 
                                gaze_direction, pitch: float, yaw: float, roll: float):
        """
        Draw gaze visualization for a single face on the image.
        
        Args:
            image: OpenCV image to draw on
            landmarks_msg: FacialLandmarks message
            gaze_score: Computed gaze confidence score
            gaze_direction: Gaze direction vector
            pitch: Head pitch angle in degrees
            yaw: Head yaw angle in degrees  
            roll: Head roll angle in degrees
        """
        try:
            # Extract face bounding box from landmarks message (now NormalizedRegionOfInterest2D)
            if hasattr(landmarks_msg.bbox_xyxy, 'xmin'):
                # bbox_xyxy is now NormalizedRegionOfInterest2D with normalized coordinates [0,1]
                # Denormalize to pixel coordinates
                x1_norm, y1_norm = landmarks_msg.bbox_xyxy.xmin, landmarks_msg.bbox_xyxy.ymin
                x2_norm, y2_norm = landmarks_msg.bbox_xyxy.xmax, landmarks_msg.bbox_xyxy.ymax
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int(x1_norm * landmarks_msg.width)
                y1 = int(y1_norm * landmarks_msg.height)
                x2 = int(x2_norm * landmarks_msg.width)
                y2 = int(y2_norm * landmarks_msg.height)
                
                # Convert to [x, y, w, h] format for visualization
                face_bbox = [x1, y1, x2 - x1, y2 - y1]
                
                # Draw gaze visualization
                self._draw_gaze_on_image(image, face_bbox, landmarks_msg,
                                      gaze_score, gaze_direction, pitch, yaw, roll)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f'Drew gaze visualization for face {landmarks_msg.face_id}')
            else:
                if self.enable_debug_output:
                    self.get_logger().debug(f'No valid bbox_xyxy for face {landmarks_msg.face_id}')
                
        except Exception as e:
            self.get_logger().error(f'Error drawing gaze visualization for face {landmarks_msg.face_id}: {str(e)}')

    def publish_gaze_visualization(self, image, landmarks_msg, gaze_score: float, 
                                 gaze_direction, pitch: float, yaw: float, roll: float):
        """
        Publish image with gaze visualization overlay.
        
        Args:
            image: OpenCV image
            landmarks_msg: FacialLandmarks message
            gaze_score: Computed gaze confidence score
            gaze_direction: Gaze direction vector
            pitch: Head pitch angle in degrees
            yaw: Head yaw angle in degrees  
            roll: Head roll angle in degrees
        """
        if not self.enable_image_output or image is None:
            if self.enable_debug_output:
                self.get_logger().debug(f'Skipping gaze visualization: enable_image_output={self.enable_image_output}, image={image is not None}')
            return
            
        try:
            # Create a copy for annotation
            annotated_image = image.copy()
            
            # Extract face bounding box from landmarks message (now NormalizedRegionOfInterest2D)
            if hasattr(landmarks_msg.bbox_xyxy, 'xmin'):
                # bbox_xyxy is now NormalizedRegionOfInterest2D with normalized coordinates [0,1]
                # Denormalize to pixel coordinates
                x1_norm, y1_norm = landmarks_msg.bbox_xyxy.xmin, landmarks_msg.bbox_xyxy.ymin
                x2_norm, y2_norm = landmarks_msg.bbox_xyxy.xmax, landmarks_msg.bbox_xyxy.ymax
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int(x1_norm * landmarks_msg.width)
                y1 = int(y1_norm * landmarks_msg.height)
                x2 = int(x2_norm * landmarks_msg.width)
                y2 = int(y2_norm * landmarks_msg.height)
                
                # Convert to [x, y, w, h] format for visualization
                face_bbox = [x1, y1, x2 - x1, y2 - y1]
                
                # Draw gaze visualization
                self._draw_gaze_on_image(annotated_image, face_bbox, landmarks_msg,
                                       gaze_score, gaze_direction, pitch, yaw, roll)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f'Drew gaze visualization for face {landmarks_msg.face_id}')
            else:
                if self.enable_debug_output:
                    self.get_logger().debug(f'No valid bbox_xyxy for face {landmarks_msg.face_id}')
            
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
