#!/usr/bin/env python3
"""
Visual Speech Activity Detection Node for ROS2

This node subscribes to FacialRecognition and FacialLandmarks messages,
performs visual speech activity detection using lip movement analysis,
and publishes extended FacialRecognition messages with speaking status.

It uses the LipMovementDetector to analyze temporal patterns in mouth
landmarks and determine if each recognized face is currently speaking.

Architecture:
- Subscribes to: FacialRecognition messages (for recognized_face_id)
- Subscribes to: FacialLandmarks messages (for mouth landmarks)
- Publishes: Extended FacialRecognition messages with is_speaking and speaking_confidence
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy
from rclpy.time import Time

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from threading import Lock
import time
import cv2
import os
from cv_bridge import CvBridge

try:
    from hri_msgs.msg import (
        FacialLandmarks, 
        FacialLandmarksArray, 
        FacialRecognition, 
        FacialRecognitionArray,
        IdsList
    )
except ImportError:
    print("Warning: hri_msgs not found. Please install hri_msgs package.")
    FacialLandmarks = None
    FacialLandmarksArray = None
    FacialRecognition = None
    FacialRecognitionArray = None

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage

from .vsdlm_detector import VSDLMDetector


def _stamp_to_float(stamp) -> float:
    """Convert ROS timestamp to float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class VisualSpeechActivityNode(Node):
    """
    ROS2 node for visual speech activity detection.
    
    This node integrates lip movement detection with face recognition results
    to provide robust speaking activity detection based on recognized_face_id.
    """
    
    def __init__(self):
        super().__init__('visual_speech_activity_node')
        
        # Declare and get parameters
        self._declare_parameters()
        self._get_parameters()
        
        # Build full VSDLM model path (similar to face_recognition approach)
        vsdlm_weights_dir_path = self._get_weights_directory_path()
        vsdlm_model_full_path = os.path.join(vsdlm_weights_dir_path, self.vsdlm_weights_name)
        self.get_logger().debug(f"VSDLM weights directory: {vsdlm_weights_dir_path}")
        self.get_logger().debug(f"VSDLM model path: {vsdlm_model_full_path}")
        
        # Initialize VSDLM detector for visual speech detection
        self.vsdlm_detector = VSDLMDetector(
            model_path=vsdlm_model_full_path,
            model_variant=self.vsdlm_model_variant,
            providers=self.vsdlm_providers,
            speaking_threshold=self.speaking_threshold,
            debug_save_crops=self.vsdlm_debug_save_crops,
            logger=self.get_logger(),
            mouth_height_ratio=self.vsdlm_mouth_height_ratio,
            temporal_smoothing=self.vsdlm_temporal_smoothing,
            smoothing_window_size=self.vsdlm_smoothing_window_size,
            min_confidence_for_change=self.vsdlm_min_confidence_for_change
        )
        
        # Log final provider selection from the detector
        actual_providers = self.vsdlm_detector.session.get_providers() if hasattr(self.vsdlm_detector, 'session') else "unknown"
        self.get_logger().debug(f"VSDLM detector initialized: variant={self.vsdlm_model_variant}, threshold={self.speaking_threshold}, "
                               f"mouth_height_ratio={self.vsdlm_mouth_height_ratio}, temporal_smoothing={self.vsdlm_temporal_smoothing}")
        self.get_logger().debug(f"VSDLM final providers: {actual_providers}")
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Image buffer: store recent images with timestamps for synchronization
        # Format: deque of (timestamp_float, cv_image_ndarray)
        # This ensures we use the SAME image that landmarks were detected from
        self.image_buffer = deque(maxlen=30)  # Keep last 30 images (~1 second at 30 fps)
        self.image_buffer_lock = Lock()
        
        # Legacy variables kept for compatibility (will be deprecated)
        self.latest_color_image_msg = None
        self.color_image_processed = False
        self.latest_color_image_timestamp = None
        
        # Message buffers for synchronization
        # Store latest facial landmarks per face_id
        self.latest_landmarks: Dict[str, FacialLandmarks] = {}
        
        # Store latest recognition per recognized_face_id
        self.latest_recognition: Dict[str, FacialRecognition] = {}
        
        # Mapping from face_id to recognized_face_id (updated from recognition messages)
        self.face_id_to_recognized_id: Dict[str, str] = {}
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.processed_messages = 0
        
        # Initialize visualization collection for multi-face images
        self.pending_visualizations = []  # List to collect face visualizations for multi-face images
        self.last_image_timestamp = None  # Track when to publish collected visualizations
        
        # Setup QoS profiles
        self.qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        
        # Sensor QoS for images (same as face_detection to avoid delays)
        self.sensor_qos = QoSProfile(
            depth=1,  # Keep only the latest image
        )
        
        # Setup subscribers and publishers
        self._setup_topics()
        
        # Periodic cleanup timer
        cleanup_period = 5.0  # Clean up old identities every 5 seconds
        self.cleanup_timer = self.create_timer(cleanup_period, self._cleanup_old_identities)
        
        # Create timer for periodic debug info if debug is enabled
        if self.enable_debug_output:
            self.debug_timer = self.create_timer(10.0, self._debug_status_callback)
        
        self.get_logger().debug("Visual Speech Activity Node initialized")
        self.get_logger().debug(f"ROS4HRI mode: {'with_id' if self.ros4hri_with_id else 'array'}")
        self.get_logger().debug(f"Face recognition mode: {'enabled' if self.use_face_recognition else 'disabled (face_id only)'}")
        self.get_logger().debug(f"VSDLM model: {self.vsdlm_model_variant}")
        self.get_logger().debug(f"Speaking threshold: {self.speaking_threshold}")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Input/Output topics
        self.declare_parameter('recognition_input_topic', '/humans/faces/recognized')
        self.declare_parameter('landmarks_input_topic', '/humans/faces/detected')
        self.declare_parameter('output_topic', '/humans/faces/speaking')
        self.declare_parameter('output_image_topic', '/humans/faces/speaking/annotated_img')
        
        # ROS4HRI mode parameter
        self.declare_parameter('ros4hri_with_id', False)  # Default to array mode
        
        # VSDLM parameters for visual speech detection
        self.declare_parameter('speaking_threshold', 0.5)  # Probability threshold for speaking classification
        self.declare_parameter('vsdlm_weights_path', 'weights')  # Directory containing VSDLM weights
        self.declare_parameter('vsdlm_weights_name', 'vsdlm_s.onnx')  # Specific VSDLM weights filename
        self.declare_parameter('vsdlm_model_variant', 'S')  # Model variant for auto-download: P/N/S/M/L
        self.declare_parameter('vsdlm_execution_provider', 'cpu')  # ONNX provider: cpu/cuda/tensorrt
        self.declare_parameter('vsdlm_mouth_height_ratio', 0.35)  # Mouth height as ratio of face height (YOLO mode)
        
        # Temporal smoothing parameters for VSDLM
        self.declare_parameter('vsdlm_temporal_smoothing', True)  # Enable temporal smoothing to reduce flickering
        self.declare_parameter('vsdlm_smoothing_window_size', 5)  # Number of frames for smoothing window
        self.declare_parameter('vsdlm_min_confidence_for_change', 0.1)  # Min confidence diff to change state
        
        # Image input parameters (same as face_recognition)
        self.declare_parameter('image_topic', '/camera/color/image_raw')  # Camera image topic
        self.declare_parameter('compressed_topic', '')  # Compressed image topic (optional)
        
        # Face recognition dependency parameter
        self.declare_parameter('use_face_recognition', True)  # Use face recognition for robust tracking
        
        # Image visualization parameters
        self.declare_parameter('enable_image_output', True)
        self.declare_parameter('label_offset_y', 50)  # Pixels above mouth center for label placement
        
        # Debug parameters
        self.declare_parameter('enable_debug_output', False)
        self.declare_parameter('vsdlm_debug_save_crops', False)
    
    def _get_parameters(self):
        """Get parameters from ROS2 parameter server."""
        # Topics
        self.recognition_input_topic = self.get_parameter('recognition_input_topic').get_parameter_value().string_value
        self.landmarks_input_topic = self.get_parameter('landmarks_input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        
        # ROS4HRI mode
        self.ros4hri_with_id = self.get_parameter('ros4hri_with_id').get_parameter_value().bool_value
        
        # VSDLM parameters
        self.speaking_threshold = self.get_parameter('speaking_threshold').get_parameter_value().double_value
        self.vsdlm_weights_path = self.get_parameter('vsdlm_weights_path').get_parameter_value().string_value
        self.vsdlm_weights_name = self.get_parameter('vsdlm_weights_name').get_parameter_value().string_value
        self.vsdlm_model_variant = self.get_parameter('vsdlm_model_variant').get_parameter_value().string_value
        vsdlm_provider = self.get_parameter('vsdlm_execution_provider').get_parameter_value().string_value
        self.vsdlm_mouth_height_ratio = self.get_parameter('vsdlm_mouth_height_ratio').get_parameter_value().double_value
        
        # Temporal smoothing parameters
        self.vsdlm_temporal_smoothing = self.get_parameter('vsdlm_temporal_smoothing').get_parameter_value().bool_value
        self.vsdlm_smoothing_window_size = self.get_parameter('vsdlm_smoothing_window_size').get_parameter_value().integer_value
        self.vsdlm_min_confidence_for_change = self.get_parameter('vsdlm_min_confidence_for_change').get_parameter_value().double_value
        
        # Image topics
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
        
        # Map provider string to ONNX provider list with intelligent fallbacks
        if vsdlm_provider == 'cuda':
            # Try CUDA first, with CPU fallback
            self.vsdlm_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif vsdlm_provider == 'tensorrt':
            # Try TensorRT, then CUDA, then CPU
            self.vsdlm_providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            # CPU-only mode
            self.vsdlm_providers = ['CPUExecutionProvider']
        
        self.get_logger().debug(f"VSDLM provider requested: {vsdlm_provider} -> providers: {self.vsdlm_providers}")
        
        # Face recognition dependency
        self.use_face_recognition = self.get_parameter('use_face_recognition').get_parameter_value().bool_value
        
        # Image visualization
        self.enable_image_output = self.get_parameter('enable_image_output').get_parameter_value().bool_value
        self.label_offset_y = self.get_parameter('label_offset_y').get_parameter_value().integer_value
        
        # Debug
        self.enable_debug_output = self.get_parameter('enable_debug_output').get_parameter_value().bool_value
        self.vsdlm_debug_save_crops = self.get_parameter('vsdlm_debug_save_crops').get_parameter_value().bool_value
        
        # Set logger level based on debug flag
        if self.enable_debug_output:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
            self.get_logger().debug("[DEBUG MODE ENABLED] Verbose logging activated")
        else:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)
    
    def _get_weights_directory_path(self) -> str:
        """
        Get the full path to the weights directory.
        Similar to face_recognition approach - navigates from package source to weights folder.
        
        Returns:
            Absolute path to the weights directory
        """
        # Get the current file's directory and navigate to package root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_src_dir = os.path.dirname(current_file_dir)
        package_src_dir = package_src_dir.replace('build', 'src')  # Save it in docker volume of ros2 package
        weights_dir_path = os.path.join(package_src_dir, self.vsdlm_weights_path)
        
        # Create weights dir if not there
        if not os.path.exists(weights_dir_path):
            os.makedirs(weights_dir_path)
            self.get_logger().debug(f"Created weights directory: {weights_dir_path}")
        
        return weights_dir_path
    
    def _setup_topics(self):
        """Setup ROS2 subscribers and publishers based on ROS4HRI mode."""
        if self.ros4hri_with_id:
            # ROS4HRI with ID mode: Per-ID topics
            # Dictionary to store subscribers and publishers for each face ID
            self.landmarks_subscribers = {}  # {face_id: Subscription}
            self.recognition_subscribers = {}  # {recognized_face_id: Subscription}
            self.speaking_publishers = {}  # {recognized_face_id: Publisher}
            self.tracked_face_ids = set()
            
            # Subscribe to tracked faces list to dynamically create subscribers
            self.tracked_faces_subscriber = self.create_subscription(
                IdsList,
                '/humans/faces/tracked',
                self._tracked_faces_callback,
                self.qos_profile
            )
            
            self.get_logger().debug("Subscribed to /humans/faces/tracked for dynamic per-ID topics")
            
            # No array-mode subscribers
            self.landmarks_array_subscriber = None
            self.recognition_array_subscriber = None
            self.speaking_array_publisher = None
        else:
            # ROS4HRI array mode: Array topics
            self.landmarks_array_subscriber = self.create_subscription(
                FacialLandmarksArray,
                self.landmarks_input_topic,
                self._landmarks_array_callback,
                self.qos_profile
            )
            
            # Only subscribe to recognition if face recognition is enabled
            if self.use_face_recognition:
                self.recognition_array_subscriber = self.create_subscription(
                    FacialRecognitionArray,
                    self.recognition_input_topic,
                    self._recognition_array_callback,
                    self.qos_profile
                )
                self.get_logger().debug(f"Subscribed to {self.recognition_input_topic} for face recognition")
            else:
                self.recognition_array_subscriber = None
                self.get_logger().debug("Face recognition disabled - working with face_id only")
            
            # Publisher
            self.speaking_array_publisher = self.create_publisher(
                FacialRecognitionArray,
                self.output_topic,
                self.qos_profile
            )
            
            self.get_logger().debug(f"Subscribed to {self.landmarks_input_topic} (array mode)")
            self.get_logger().debug(f"Publishing to {self.output_topic} (array mode)")
            
            # No per-ID structures
            self.landmarks_subscribers = {}
            self.recognition_subscribers = {}
            self.speaking_publishers = {}
            self.tracked_faces_subscriber = None
        
        # Subscribe to camera image (same pattern as face_recognition)
        if self.compressed_topic and self.compressed_topic.strip():
            self.get_logger().debug(f"Using compressed image topic: {self.compressed_topic}")
            real_time_qos = QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE
            )
            self.color_sub = self.create_subscription(
                CompressedImage,
                self.compressed_topic,
                self._store_latest_compressed_rgb,
                real_time_qos
            )
        else:
            self.get_logger().debug(f"Using regular image topic: {self.image_topic}")
            self.color_sub = self.create_subscription(
                Image,
                self.image_topic,
                self._store_latest_rgb,
                self.sensor_qos
            )
        
        # Create image publisher for visualization
        self.image_publisher = None
        if self.enable_image_output:
            self.image_publisher = self.create_publisher(
                Image,
                self.output_image_topic,
                10
            )
            self.get_logger().debug(f"Publishing annotated images to {self.output_image_topic}")
    
    def _store_latest_rgb(self, color_msg: Image):
        """Store latest color image with timestamp in buffer."""
        try:
            # Convert to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            timestamp = _stamp_to_float(color_msg.header.stamp)
            
            # Add to timestamped buffer
            with self.image_buffer_lock:
                self.image_buffer.append((timestamp, cv_image))
            
            # Keep legacy variables for compatibility
            self.latest_color_image_msg = color_msg
            self.color_image_processed = False
            self.latest_color_image_timestamp = self.get_clock().now()
            
            if self.enable_debug_output and not hasattr(self, '_image_received_logged'):
                self.get_logger().debug(f"[NODE-IMAGE] First image received on topic {self.image_topic}, size: {color_msg.width}x{color_msg.height}")
                self._image_received_logged = True
        except Exception as e:
            self.get_logger().error(f"Error storing RGB image: {e}")
    
    def _store_latest_compressed_rgb(self, color_msg: CompressedImage):
        """Store latest compressed color image with timestamp in buffer."""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(color_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error('Failed to decode compressed image')
                return
            
            timestamp = _stamp_to_float(color_msg.header.stamp)
            
            # Add to timestamped buffer
            with self.image_buffer_lock:
                self.image_buffer.append((timestamp, cv_image))
            
            # Keep legacy variables for compatibility
            self.latest_color_image_msg = color_msg
            self.color_image_processed = False
            self.latest_color_image_timestamp = self.get_clock().now()
            
            if self.enable_debug_output and not hasattr(self, '_compressed_image_received_logged'):
                self.get_logger().debug(f"[NODE-IMAGE] First compressed image received on topic {self.compressed_topic}")
                self._compressed_image_received_logged = True
        except Exception as e:
            self.get_logger().error(f"Error storing compressed image: {e}")
    
    def _get_latest_image(self) -> Optional[np.ndarray]:
        """Get latest image (legacy method for backward compatibility)."""
        with self.image_buffer_lock:
            if len(self.image_buffer) == 0:
                return None
            # Return most recent image
            return self.image_buffer[-1][1]
    
    def _get_image_by_timestamp(self, target_timestamp: float, slop: float = 0.1) -> Optional[np.ndarray]:
        """
        Get image that matches the target timestamp within slop tolerance.
        
        This ensures landmarks are drawn on the SAME image they were detected from.
        
        Args:
            target_timestamp: Target timestamp in seconds (from landmarks header)
            slop: Maximum time difference in seconds (default 100ms)
            
        Returns:
            BGR image as numpy array or None if no matching image found
        """
        with self.image_buffer_lock:
            if len(self.image_buffer) == 0:
                if self.enable_debug_output and not hasattr(self, '_no_image_warned'):
                    self.get_logger().warn(f"[NODE-IMAGE] No images in buffer yet")
                    self._no_image_warned = True
                return None
            
            # Find image with closest timestamp within slop
            best_image = None
            best_dt = float('inf')
            
            for img_timestamp, cv_image in self.image_buffer:
                dt = abs(img_timestamp - target_timestamp)
                if dt < best_dt:
                    best_dt = dt
                    best_image = cv_image
            
            # Check if best match is within slop tolerance
            if best_dt > slop:
                if self.enable_debug_output:
                    self.get_logger().warn(
                        f"[NODE-IMAGE] No image within {slop*1000:.0f}ms slop for timestamp {target_timestamp:.3f}. "
                        f"Closest: {best_dt*1000:.1f}ms away"
                    )
                return None
            
            if self.enable_debug_output:
                self.get_logger().debug(
                    f"[NODE-IMAGE] Matched image with dt={best_dt*1000:.1f}ms for timestamp {target_timestamp:.3f}"
                )
            
            return best_image
    
    # -------------------------------------------------------------------------
    #                    ROS4HRI with ID Mode Callbacks
    # -------------------------------------------------------------------------
    
    def _tracked_faces_callback(self, msg: IdsList):
        """
        Callback for tracked faces list in ROS4HRI with ID mode.
        
        Creates/removes per-ID subscribers and publishers based on active face IDs.
        """
        current_ids = set(msg.ids)
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Tracked faces callback: {list(current_ids)}")
        
        # Add new face IDs
        for face_id in current_ids:
            if face_id not in self.tracked_face_ids:
                self._create_per_id_subscribers(face_id)
                self.tracked_face_ids.add(face_id)
                self.get_logger().debug(f"Added tracked face: {face_id}")
        
        # Remove old face IDs
        removed_ids = self.tracked_face_ids - current_ids
        for face_id in removed_ids:
            self._remove_per_id_subscribers(face_id)
            self.tracked_face_ids.discard(face_id)
            self.get_logger().debug(f"Removed tracked face: {face_id}")
    
    def _create_per_id_subscribers(self, face_id: str):
        """Create per-ID subscribers for a new tracked face."""
        # Subscribe to landmarks for this face - use 'detected' not 'landmarks'
        landmarks_topic = f'/humans/faces/{face_id}/detected'
        self.landmarks_subscribers[face_id] = self.create_subscription(
            FacialLandmarks,
            landmarks_topic,
            lambda msg, fid=face_id: self._landmarks_per_id_callback(msg, fid),
            self.qos_profile
        )
        
        self.get_logger().debug(f"Created subscriber for topic: {landmarks_topic}")
        if self.enable_debug_output:
            self.get_logger().debug(f"Subscribed to {landmarks_topic} with QoS profile")
        
        # If face recognition is enabled, also subscribe to recognition topic for this face
        if self.use_face_recognition:
            self._create_per_id_recognition_subscriber(face_id)
    
    def _remove_per_id_subscribers(self, face_id: str):
        """Remove per-ID subscribers for a face that's no longer tracked."""
        if face_id in self.landmarks_subscribers:
            self.destroy_subscription(self.landmarks_subscribers[face_id])
            del self.landmarks_subscribers[face_id]
        
        if face_id in self.recognition_subscribers:
            self.destroy_subscription(self.recognition_subscribers[face_id])
            del self.recognition_subscribers[face_id]
        
        # Clean up speaking publishers too
        if face_id in self.speaking_publishers:
            self.destroy_publisher(self.speaking_publishers[face_id])
            del self.speaking_publishers[face_id]
        
        if face_id in self.latest_landmarks:
            del self.latest_landmarks[face_id]
    
    def _landmarks_per_id_callback(self, msg: FacialLandmarks, face_id: str):
        """Callback for per-ID facial landmarks."""
        if self.enable_debug_output:
            self.get_logger().debug(f"Landmarks callback triggered for face {face_id}")
        
        self.latest_landmarks[face_id] = msg
        
        if self.use_face_recognition:
            # Process immediately if we have corresponding recognition
            if face_id in self.face_id_to_recognized_id:
                recognized_id = self.face_id_to_recognized_id[face_id]
                if recognized_id in self.latest_recognition:
                    self._process_single_face(face_id, recognized_id)
        else:
            # Process landmarks directly without face recognition
            self._process_single_landmark_without_recognition(msg)
    
    def _create_per_id_recognition_subscriber(self, face_id: str):
        """Create per-ID subscriber for recognition messages."""
        if face_id not in self.recognition_subscribers:
            recognition_topic = f'/humans/faces/{face_id}/recognized'
            self.recognition_subscribers[face_id] = self.create_subscription(
                FacialRecognition,
                recognition_topic,
                lambda msg, fid=face_id: self._recognition_per_id_callback(msg, fid),
                self.qos_profile
            )
            
            if self.enable_debug_output:
                self.get_logger().debug(f"Subscribed to {recognition_topic}")
    
    def _recognition_per_id_callback(self, msg: FacialRecognition, face_id: str):
        """Callback for per-ID facial recognition."""
        recognized_face_id = msg.recognized_face_id
        self.latest_recognition[recognized_face_id] = msg
        self.face_id_to_recognized_id[msg.face_id] = recognized_face_id
        
        # Process if we have landmarks
        if msg.face_id in self.latest_landmarks:
            self._process_single_face(msg.face_id, recognized_face_id)
    
    # -------------------------------------------------------------------------
    #                    ROS4HRI Array Mode Callbacks
    # -------------------------------------------------------------------------
    
    def _landmarks_array_callback(self, msg: FacialLandmarksArray):
        """Callback for facial landmarks array."""
        # Store latest landmarks for each face
        for landmarks in msg.ids:
            self.latest_landmarks[landmarks.face_id] = landmarks
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Received {len(msg.ids)} facial landmarks")
        
        # If face recognition is disabled, process landmarks directly
        if not self.use_face_recognition:
            self._process_landmarks_without_recognition(msg.ids)
    
    def _recognition_array_callback(self, msg: FacialRecognitionArray):
        """
        Callback for facial recognition array.
        
        This triggers the speaking detection process for all recognized faces.
        """
        start_time = time.time()
        
        if len(msg.facial_recognition) == 0:
            # Publish empty result
            self._publish_speaking_array([])
            return
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Processing {len(msg.facial_recognition)} recognition(s)")
        
        # Update mappings and process each recognition
        speaking_recognitions = []
        
        for recognition in msg.facial_recognition:
            # Update mapping
            self.latest_recognition[recognition.recognized_face_id] = recognition
            self.face_id_to_recognized_id[recognition.face_id] = recognition.recognized_face_id
            
            # Process speaking detection
            speaking_recognition = self._process_recognition(recognition)
            speaking_recognitions.append(speaking_recognition)
        
        # Publish results
        self._publish_speaking_array(speaking_recognitions)
        
        # Track performance
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.processed_messages += 1
        
        if self.enable_debug_output and self.processed_messages % 30 == 0:
            avg_time = self.total_processing_time / self.processed_messages
            self.get_logger().debug(f"Avg processing time: {avg_time*1000:.2f}ms")
    
    # -------------------------------------------------------------------------
    #                    Core Processing Methods
    # -------------------------------------------------------------------------
    
    def _process_recognition(self, recognition: FacialRecognition) -> FacialRecognition:
        """
        Process a single recognition message and add speaking detection.
        
        Args:
            recognition: Input FacialRecognition message
            
        Returns:
            Extended FacialRecognition message with speaking status
        """
        # Get corresponding landmarks
        landmarks_msg = self.latest_landmarks.get(recognition.face_id)
        
        if landmarks_msg is None:
            # No landmarks available, return recognition with default speaking status
            extended_recognition = self._copy_recognition_with_speaking(
                recognition, is_speaking=False, speaking_confidence=0.0
            )
            if self.enable_debug_output:
                self.get_logger().debug(
                    f"No landmarks for face_id {recognition.face_id}, defaulting to not speaking"
                )
            return extended_recognition
        
        # Convert landmarks to list of (x, y, c) tuples
        landmarks_list = self._extract_landmark_coordinates(landmarks_msg)
        
        # Extract face bounding box for YOLO landmark support
        face_bbox = self._extract_face_bbox(landmarks_msg)
        
        # Get image that matches the landmarks timestamp (within 100ms slop)
        landmarks_timestamp = _stamp_to_float(landmarks_msg.header.stamp)
        cv_image = self._get_image_by_timestamp(landmarks_timestamp, slop=0.1)
        
        # Detect speaking using VSDLM
        if cv_image is None:
            if self.enable_debug_output:
                self.get_logger().warning(
                    f"No synchronized image for landmarks at t={landmarks_timestamp:.3f} "
                    f"for face {recognition.recognized_face_id}"
                )
            is_speaking, speaking_confidence, mouth_crop_bbox = False, 0.0, None
        else:
            if self.enable_debug_output:
                self.get_logger().debug(
                    f"[NODE] Using synchronized image (t={landmarks_timestamp:.3f}) for face {recognition.recognized_face_id}, "
                    f"image shape: {cv_image.shape}, landmarks count: {len(landmarks_list)}, face_bbox: {face_bbox}"
                )
            
            is_speaking, speaking_confidence, mouth_crop_bbox = self.vsdlm_detector.detect_speaking(
                cv_image,
                landmarks_list,
                face_bbox,
                face_id=recognition.recognized_face_id
            )
            
            if self.enable_debug_output:
                self.get_logger().debug(f"[NODE] VSDLM returned: is_speaking={is_speaking}, speaking_confidence={speaking_confidence}, mouth_bbox={mouth_crop_bbox}")
        
        # Create extended recognition message
        extended_recognition = self._copy_recognition_with_speaking(
            recognition, is_speaking, speaking_confidence
        )
        
        # Publish visualization if enabled and image available
        if self.enable_image_output and self.image_publisher is not None and cv_image is not None:
            self._collect_face_visualization(
                cv_image, landmarks_list, is_speaking, speaking_confidence,
                recognition.recognized_face_id, landmarks_msg.header, mouth_crop_bbox
            )
        
        if self.enable_debug_output:
            ext_is_spk = extended_recognition.is_speaking if hasattr(extended_recognition, 'is_speaking') else 'N/A'
            ext_spk_conf = extended_recognition.speaking_confidence if hasattr(extended_recognition, 'speaking_confidence') else 'N/A'
            self.get_logger().debug(
                f"[NODE] Face {recognition.recognized_face_id}: "
                f"speaking={is_speaking}, confidence={speaking_confidence:.4f} -> "
                f"extended_recognition.is_speaking={ext_is_spk}, "
                f"extended_recognition.speaking_confidence={ext_spk_conf}"
            )
        
        return extended_recognition
    
    def _process_single_face(self, face_id: str, recognized_face_id: str):
        """
        Process a single face in per-ID mode.
        
        Args:
            face_id: Volatile face tracking ID
            recognized_face_id: Stable recognized face ID
        """
        recognition = self.latest_recognition.get(recognized_face_id)
        if recognition is None:
            return
        
        # Process and publish
        speaking_recognition = self._process_recognition(recognition)
        self._publish_speaking_per_id(speaking_recognition, face_id)
    
    def _process_landmarks_without_recognition(self, landmarks_list: List[FacialLandmarks]):
        """
        Process landmarks directly when face recognition is disabled.
        
        Uses face_id as the identity for tracking instead of recognized_face_id.
        
        Args:
            landmarks_list: List of FacialLandmarks messages
        """
        speaking_recognitions = []
        
        for landmarks_msg in landmarks_list:
            face_id = landmarks_msg.face_id
            
            # Convert landmarks to coordinate list
            landmarks_coords = self._extract_landmark_coordinates(landmarks_msg)
            
            # Extract face bounding box for YOLO landmark support
            face_bbox = self._extract_face_bbox(landmarks_msg)
            
            # Get image that matches the landmarks timestamp
            landmarks_timestamp = _stamp_to_float(landmarks_msg.header.stamp)
            cv_image = self._get_image_by_timestamp(landmarks_timestamp, slop=0.1)
            
            if self.enable_debug_output:
                self.get_logger().debug(
                    f"[NODE-ARRAY] Processing face {face_id} at t={landmarks_timestamp:.3f}, "
                    f"image available: {cv_image is not None}, landmarks: {len(landmarks_coords)}, face_bbox: {face_bbox}"
                )
            
            # Detect speaking using VSDLM
            if cv_image is None:
                is_speaking, speaking_confidence, mouth_crop_bbox = False, 0.0, None
            else:
                is_speaking, speaking_confidence, mouth_crop_bbox = self.vsdlm_detector.detect_speaking(
                    cv_image,
                    landmarks_coords,
                    face_bbox,
                    face_id=face_id
                )
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"[NODE-ARRAY] VSDLM returned for {face_id}: is_speaking={is_speaking}, confidence={speaking_confidence:.4f}, mouth_bbox={mouth_crop_bbox}")
            
            # Publish visualization if enabled and image available
            if self.enable_image_output and self.image_publisher is not None and cv_image is not None:
                self._collect_face_visualization(
                    cv_image, landmarks_coords, is_speaking, speaking_confidence,
                    face_id, landmarks_msg.header, mouth_crop_bbox
                )
            
            # Create a FacialRecognition message using face_id
            recognition = FacialRecognition()
            recognition.header = landmarks_msg.header
            recognition.face_id = face_id
            recognition.recognized_face_id = face_id  # Same as face_id when no recognition
            recognition.confidence = 1.0  # Full confidence since no recognition uncertainty
            
            # Set speaking fields if they exist (requires rebuilt hri_msgs)
            if hasattr(recognition, 'is_speaking'):
                recognition.is_speaking = is_speaking
                recognition.speaking_confidence = speaking_confidence
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"[NODE-ARRAY] Set speaking fields: is_speaking={recognition.is_speaking}, speaking_confidence={recognition.speaking_confidence}")
            
            speaking_recognitions.append(recognition)
            
            if self.enable_debug_output:
                self.get_logger().debug(
                    f"Face {face_id} (no recognition): "
                    f"speaking={is_speaking}, confidence={speaking_confidence:.3f}"
                )
        
        # Publish results
        self._publish_speaking_array(speaking_recognitions)
    
    def _process_single_landmark_without_recognition(self, landmarks_msg: FacialLandmarks):
        """
        Process a single landmark message without face recognition in per-ID mode.
        
        Args:
            landmarks_msg: FacialLandmarks message
        """
        face_id = landmarks_msg.face_id
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Received landmarks for face {face_id}, landmarks count: {len(landmarks_msg.landmarks)}")
        
        # Convert landmarks to coordinate list
        landmarks_coords = self._extract_landmark_coordinates(landmarks_msg)
        
        # Extract face bounding box for YOLO landmark support
        face_bbox = self._extract_face_bbox(landmarks_msg)
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Extracted {len(landmarks_coords)} landmark coordinates for face {face_id}, bbox: {face_bbox}")
        
        # Get image that matches the landmarks timestamp
        landmarks_timestamp = _stamp_to_float(landmarks_msg.header.stamp)
        cv_image = self._get_image_by_timestamp(landmarks_timestamp, slop=0.1)
        
        if self.enable_debug_output:
            self.get_logger().debug(
                f"[NODE-PERID] Processing face {face_id} at t={landmarks_timestamp:.3f}, "
                f"image available: {cv_image is not None}"
            )
        
        # Detect speaking using VSDLM
        if cv_image is None:
            is_speaking, speaking_confidence, mouth_crop_bbox = False, 0.0, None
        else:
            is_speaking, speaking_confidence, mouth_crop_bbox = self.vsdlm_detector.detect_speaking(
                cv_image,
                landmarks_coords,
                face_bbox,
                face_id=face_id
            )
            
            if self.enable_debug_output:
                self.get_logger().debug(f"[NODE-PERID] VSDLM returned: is_speaking={is_speaking}, confidence={speaking_confidence:.4f}, mouth_bbox={mouth_crop_bbox}")
        
        # Publish visualization if enabled and image available
        if self.enable_image_output and self.image_publisher is not None and cv_image is not None:
            self._collect_face_visualization(
                cv_image, landmarks_coords, is_speaking, speaking_confidence,
                face_id, landmarks_msg.header, mouth_crop_bbox
            )
        
        # Create a FacialRecognition message using face_id
        recognition = FacialRecognition()
        recognition.header = landmarks_msg.header
        recognition.face_id = face_id
        recognition.recognized_face_id = face_id  # Same as face_id when no recognition
        recognition.confidence = 1.0  # Full confidence since no recognition uncertainty
        
        # Set speaking fields if they exist (requires rebuilt hri_msgs)
        if hasattr(recognition, 'is_speaking'):
            recognition.is_speaking = is_speaking
            recognition.speaking_confidence = speaking_confidence
            
            if self.enable_debug_output:
                self.get_logger().debug(f"[NODE-PERID] Message fields set: is_speaking={recognition.is_speaking}, speaking_confidence={recognition.speaking_confidence}")
        else:
            if not hasattr(self, '_missing_fields_warned'):
                self.get_logger().error(
                    "[NODE-PERID] FacialRecognition message missing 'is_speaking' and 'speaking_confidence' fields! "
                    "You need to REBUILD hri_msgs package. Available fields: " + str([f for f in dir(recognition) if not f.startswith('_')])
                )
                self._missing_fields_warned = True
        
        # Publish per-ID result
        self._publish_speaking_per_id(recognition, face_id)
        
        if self.enable_debug_output:
            self.get_logger().debug(
                f"[NODE-PERID] Published for face {face_id}: "
                f"speaking={is_speaking}, confidence={speaking_confidence:.4f}"
            )
    
    def _debug_status_callback(self):
        """Periodic debug status callback."""
        if not self.enable_debug_output:
            return
        
        if self.ros4hri_with_id:
            # Per-ID mode debug info
            self.get_logger().debug(f"Active tracked faces: {list(self.tracked_face_ids)}")
            self.get_logger().debug(f"Active landmark subscribers: {list(self.landmarks_subscribers.keys())}")
        else:
            # Array mode debug info
            self.get_logger().debug("Running in array mode")
            
        self.get_logger().debug(f"Latest landmarks cache: {list(self.latest_landmarks.keys())}")
        
        # Show detector status
        self.get_logger().debug(f"Using VSDLM detector (variant: {self.vsdlm_model_variant})")
    
    def _extract_landmark_coordinates(self, landmarks_msg: FacialLandmarks) -> List[Tuple[float, float, float]]:
        """
        Extract landmark coordinates from FacialLandmarks message.
        
        Converts normalized coordinates to pixel coordinates for processing.
        
        Args:
            landmarks_msg: FacialLandmarks message
            
        Returns:
            List of (x, y, c) tuples in pixel coordinates with confidence
        """
        width = landmarks_msg.width
        height = landmarks_msg.height
        
        coordinates = []
        for landmark in landmarks_msg.landmarks:
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * width
            y = landmark.y * height
            c = landmark.c  # Confidence value
            coordinates.append((x, y, c))
        
        return coordinates
    
    def _extract_face_bbox(self, landmarks_msg: FacialLandmarks) -> Optional[Tuple[float, float, float, float]]:
        """
        Extract face bounding box from FacialLandmarks message.
        
        Args:
            landmarks_msg: FacialLandmarks message
            
        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in normalized coordinates or None
        """
        if hasattr(landmarks_msg, 'bbox_xyxy'):
            bbox = landmarks_msg.bbox_xyxy
            return (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        return None
    
    def _copy_recognition_with_speaking(
        self, 
        recognition: FacialRecognition,
        is_speaking: bool,
        speaking_confidence: float
    ) -> FacialRecognition:
        """
        Create a copy of FacialRecognition message with speaking fields added.
        
        Args:
            recognition: Original FacialRecognition message
            is_speaking: Speaking detection result
            speaking_confidence: Speaking confidence score
            
        Returns:
            New FacialRecognition message with speaking fields
        """
        if self.enable_debug_output:
            self.get_logger().debug(f"[NODE-COPY] Input values: is_speaking={is_speaking} (type={type(is_speaking)}), speaking_confidence={speaking_confidence} (type={type(speaking_confidence)})")
        
        extended_recognition = FacialRecognition()
        extended_recognition.header = recognition.header
        extended_recognition.face_id = recognition.face_id
        extended_recognition.recognized_face_id = recognition.recognized_face_id
        extended_recognition.confidence = recognition.confidence
        
        # Add speaking fields if they exist (requires rebuilt hri_msgs)
        if hasattr(extended_recognition, 'is_speaking'):
            # Convert numpy types to Python native types to avoid ROS message type issues
            extended_recognition.is_speaking = bool(is_speaking)
            extended_recognition.speaking_confidence = float(speaking_confidence)
            
            if self.enable_debug_output:
                self.get_logger().debug(f"[NODE-COPY] After assignment: extended_recognition.is_speaking={extended_recognition.is_speaking}, extended_recognition.speaking_confidence={extended_recognition.speaking_confidence}")
        else:
            if self.enable_debug_output:
                self.get_logger().warn("[NODE-COPY] FacialRecognition message missing speaking fields - hri_msgs needs rebuild!")
        
        return extended_recognition
    
    # -------------------------------------------------------------------------
    #                    Publishing Methods
    # -------------------------------------------------------------------------
    
    def _publish_speaking_array(self, speaking_recognitions: List[FacialRecognition]):
        """Publish speaking detection results as array."""
        msg = FacialRecognitionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.facial_recognition = speaking_recognitions
        
        if self.enable_debug_output:
            for idx, r in enumerate(speaking_recognitions):
                is_spk = r.is_speaking if hasattr(r, 'is_speaking') else 'N/A'
                spk_conf = r.speaking_confidence if hasattr(r, 'speaking_confidence') else 'N/A'
                self.get_logger().debug(
                    f"[NODE-PUBLISH-ARRAY] Recognition[{idx}]: face_id={r.face_id}, "
                    f"is_speaking={is_spk}, speaking_confidence={spk_conf}"
                )
        
        self.speaking_array_publisher.publish(msg)
        
        if self.enable_debug_output:
            speaking_count = sum(1 for r in speaking_recognitions if hasattr(r, 'is_speaking') and r.is_speaking)
            self.get_logger().debug(
                f"[NODE-PUBLISH-ARRAY] Published {len(speaking_recognitions)} recognitions, "
                f"{speaking_count} speaking"
            )
    
    def _publish_speaking_per_id(self, speaking_recognition: FacialRecognition, face_id: str):
        """Publish speaking detection result for a single face ID."""
        # Create publisher if it doesn't exist - use face_id for topic naming
        if face_id not in self.speaking_publishers:
            topic = f'/humans/faces/{face_id}/speaking'
            self.speaking_publishers[face_id] = self.create_publisher(
                FacialRecognition,
                topic,
                self.qos_profile
            )
            if self.enable_debug_output:
                self.get_logger().debug(f"[NODE-PUBLISH-PERID] Created publisher for topic: {topic}")
        
        if self.enable_debug_output:
            is_spk = speaking_recognition.is_speaking if hasattr(speaking_recognition, 'is_speaking') else 'N/A'
            spk_conf = speaking_recognition.speaking_confidence if hasattr(speaking_recognition, 'speaking_confidence') else 'N/A'
            self.get_logger().debug(
                f"[NODE-PUBLISH-PERID] Publishing to {face_id}: "
                f"is_speaking={is_spk}, speaking_confidence={spk_conf}"
            )
        
        self.speaking_publishers[face_id].publish(speaking_recognition)
    
    # -------------------------------------------------------------------------
    #                    Cleanup Methods
    # -------------------------------------------------------------------------
    
    def _cleanup_old_identities(self):
        """Periodic cleanup of old identities from buffers."""
        # Clean up local caches (keep last 100 recognitions to avoid unbounded growth)
        if len(self.latest_recognition) > 100:
            # Sort by last access (we don't track this, so just keep most recent)
            # In a production system, you'd want to track last access time
            items = list(self.latest_recognition.items())
            self.latest_recognition = dict(items[-100:])
        
        # Clean up VSDLM detector temporal buffers
        # Get currently active face IDs from various sources
        active_face_ids = set()
        
        # Add face_ids from latest landmarks
        active_face_ids.update(self.latest_landmarks.keys())
        
        # Add recognized_face_ids from latest recognition
        active_face_ids.update(self.latest_recognition.keys())
        
        # Add face_ids from face_id_to_recognized_id mapping
        active_face_ids.update(self.face_id_to_recognized_id.keys())
        active_face_ids.update(self.face_id_to_recognized_id.values())
        
        # Clean up VSDLM temporal buffers
        if active_face_ids:
            self.vsdlm_detector.cleanup_old_identities(list(active_face_ids))
        
        if self.enable_debug_output and len(active_face_ids) > 0:
            self.get_logger().debug(f"[CLEANUP] Active face IDs: {active_face_ids}")
    
    # -------------------------------------------------------------------------
    #                    Visualization Methods
    # -------------------------------------------------------------------------
    
    def _collect_face_visualization(
        self,
        cv_image: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
        is_speaking: bool,
        speaking_confidence: float,
        face_id: str,
        header: Header,
        mouth_crop_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Collect face visualization data for later batch processing.
        
        This method collects data for multiple faces and publishes a single
        annotated image when all faces for a timestamp are processed.
        
        Args:
            cv_image: Original OpenCV image
            landmarks: List of (x, y, c) landmark coordinates with confidence
            is_speaking: Speaking detection result
            speaking_confidence: Speaking confidence score
            face_id: Face identifier for labeling
            header: Original image header
            mouth_crop_bbox: Optional mouth crop bbox (x1, y1, x2, y2) for visualization
        """
        current_timestamp = header.stamp.sec + header.stamp.nanosec / 1e9
        
        # If this is a new timestamp, publish previous batch and start new one
        if (self.last_image_timestamp is not None and 
            abs(current_timestamp - self.last_image_timestamp) > 0.1):  # 100ms tolerance
            self._publish_collected_visualizations()
        
        # Add current face to visualization collection
        self.pending_visualizations.append({
            'cv_image': cv_image.copy(),
            'landmarks': landmarks,
            'is_speaking': is_speaking,
            'speaking_confidence': speaking_confidence,
            'face_id': face_id,
            'header': header,
            'mouth_crop_bbox': mouth_crop_bbox
        })
        
        self.last_image_timestamp = current_timestamp
        
        # Set a small timer to publish if no more faces come
        # This ensures we don't wait forever for additional faces
        self._schedule_visualization_publish()
    
    def _schedule_visualization_publish(self):
        """Schedule visualization publishing after a short delay."""
        # Use a timer to publish after 50ms if no new faces arrive
        if hasattr(self, '_viz_timer'):
            self._viz_timer.cancel()
        
        self._viz_timer = self.create_timer(0.05, self._publish_collected_visualizations_callback)
    
    def _publish_collected_visualizations_callback(self):
        """Timer callback to publish collected visualizations."""
        if hasattr(self, '_viz_timer'):
            self._viz_timer.cancel()
        self._publish_collected_visualizations()
    
    def _publish_collected_visualizations(self):
        """Publish all collected face visualizations on a single image."""
        if not self.pending_visualizations:
            return
            
        try:
            # Use the image from the first face (they should all be the same)
            base_data = self.pending_visualizations[0]
            annotated_image = base_data['cv_image'].copy()
            header = base_data['header']
            
            # Process all faces
            for face_data in self.pending_visualizations:
                self._draw_face_visualization(
                    annotated_image,
                    face_data['landmarks'],
                    face_data['is_speaking'],
                    face_data['speaking_confidence'],
                    face_data['face_id'],
                    face_data['mouth_crop_bbox']
                )
            
            # Publish the final annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = header
            self.image_publisher.publish(annotated_msg)
            
            if self.enable_debug_output:
                face_ids = [data['face_id'] for data in self.pending_visualizations]
                self.get_logger().debug(
                    f"Published batch visualization for {len(face_ids)} faces: {face_ids}"
                )
            
            # Clear the collection
            self.pending_visualizations.clear()
            
        except Exception as e:
            self.get_logger().error(f"Error publishing collected visualizations: {e}")
            self.pending_visualizations.clear()
    
    def _draw_face_visualization(
        self,
        annotated_image: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
        is_speaking: bool,
        speaking_confidence: float,
        face_id: str,
        mouth_crop_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Draw visualization for a single face on the provided image.
        
        Args:
            annotated_image: Image to draw on (modified in place)
            landmarks: List of (x, y, c) landmark coordinates with confidence
            is_speaking: Speaking detection result
            speaking_confidence: Speaking confidence score
            face_id: Face identifier for labeling
            mouth_crop_bbox: Optional mouth crop bbox (x1, y1, x2, y2) for visualization
        """
        # Detect if we have YOLO or dlib landmarks
        valid_landmarks = sum(1 for lm in landmarks if len(lm) >= 3 and lm[2] > 0.0)
        is_yolo = valid_landmarks < 10
        
        # Calculate adaptive sizes based on image dimensions
        img_height, img_width = annotated_image.shape[:2]
        base_size = min(img_width, img_height)
        
        adaptive_line_thickness = max(2, int(base_size * 0.003))  # 0.3% of image size
        adaptive_landmark_radius = max(3, int(base_size * 0.008))  # 0.8% of image size
        adaptive_font_scale = max(0.6, base_size * 0.0012)  # Adaptive font size
        adaptive_font_thickness = max(2, int(base_size * 0.002))  # Adaptive font thickness
        
        # Choose color based on speaking status
        if is_speaking:
            mouth_color = (0, 255, 0)  # Green for speaking
            status_text = "SPEAKING"
        else:
            mouth_color = (0, 0, 255)  # Red for not speaking
            status_text = "NOT SPEAKING"
        
        # Visualization based on landmark type
        if is_yolo:
            # YOLO mode: Draw the crop rectangle and the 2 mouth landmarks
            if mouth_crop_bbox is not None:
                x1, y1, x2, y2 = mouth_crop_bbox
                # Draw thick rectangle for the mouth crop area
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), mouth_color, adaptive_line_thickness * 2)
                # Add label on the rectangle
                cv2.putText(annotated_image, "MOUTH CROP", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale * 0.7, mouth_color, adaptive_font_thickness)
            
            # Draw YOLO mouth corner landmarks
            if len(landmarks) > 48 and landmarks[48][2] > 0:  # Left mouth
                pt = (int(landmarks[48][0]), int(landmarks[48][1]))
                cv2.circle(annotated_image, pt, adaptive_landmark_radius * 2, (255, 0, 0), -1)  # Blue
                cv2.putText(annotated_image, "L", (pt[0] + 10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, (255, 0, 0), adaptive_font_thickness)
            
            if len(landmarks) > 54 and landmarks[54][2] > 0:  # Right mouth
                pt = (int(landmarks[54][0]), int(landmarks[54][1]))
                cv2.circle(annotated_image, pt, adaptive_landmark_radius * 2, (255, 0, 0), -1)  # Blue
                cv2.putText(annotated_image, "R", (pt[0] + 10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, (255, 0, 0), adaptive_font_thickness)
            
            # Calculate label position from mouth crop bbox or landmarks
            if mouth_crop_bbox is not None:
                mouth_center_x = (mouth_crop_bbox[0] + mouth_crop_bbox[2]) // 2
                mouth_center_y = (mouth_crop_bbox[1] + mouth_crop_bbox[3]) // 2
            elif len(landmarks) > 54 and landmarks[48][2] > 0 and landmarks[54][2] > 0:
                mouth_center_x = int((landmarks[48][0] + landmarks[54][0]) / 2)
                mouth_center_y = int((landmarks[48][1] + landmarks[54][1]) / 2)
            else:
                mouth_center_x = img_width // 2
                mouth_center_y = img_height // 2
        else:
            # dlib mode: Draw full mouth contours
            if len(landmarks) < 68:
                if self.enable_debug_output:
                    self.get_logger().warn(f"Expected 68 landmarks for dlib visualization, got {len(landmarks)}. Skipping.")
                return
            
            # Draw mouth outer contour (landmarks 48-59)
            for i in range(48, 59):
                pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
                pt2 = (int(landmarks[i + 1][0]), int(landmarks[i + 1][1]))
                cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
            
            # Close mouth outer contour
            pt1 = (int(landmarks[59][0]), int(landmarks[59][1]))
            pt2 = (int(landmarks[48][0]), int(landmarks[48][1]))
            cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
            
            # Draw mouth inner contour (landmarks 60-67)
            for i in range(60, 67):
                pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
                pt2 = (int(landmarks[i + 1][0]), int(landmarks[i + 1][1]))
                cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
            
            # Close mouth inner contour
            pt1 = (int(landmarks[67][0]), int(landmarks[67][1]))
            pt2 = (int(landmarks[60][0]), int(landmarks[60][1]))
            cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
            
            # Draw mouth landmarks as circles
            for i in range(48, 68):  # Mouth landmarks (48-67)
                lm_x, lm_y = int(landmarks[i][0]), int(landmarks[i][1])
                cv2.circle(annotated_image, (lm_x, lm_y), adaptive_landmark_radius, mouth_color, -1)
            
            # Calculate mouth center for label placement
            mouth_landmarks = landmarks[48:68]
            mouth_center_x = int(np.mean([lm[0] for lm in mouth_landmarks]))
            mouth_center_y = int(np.mean([lm[1] for lm in mouth_landmarks]))
        
        # Format speaking threshold to first 2 digits (e.g., 0.5 -> "0.5")
        threshold_str = f"{self.speaking_threshold:.1f}"
        
        # Create label with status, confidence, and threshold
        label = f"Face {face_id}: {status_text} ({speaking_confidence:.2f})"
        threshold_label = f"Threshold: {threshold_str}"
        
        # Draw label background and text ABOVE mouth using configurable offset
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                    adaptive_font_scale, adaptive_font_thickness)[0]
        threshold_size = cv2.getTextSize(threshold_label, cv2.FONT_HERSHEY_SIMPLEX,
                                        adaptive_font_scale * 0.8, adaptive_font_thickness)[0]
        
        label_padding = max(5, int(base_size * 0.01))
        label_x = mouth_center_x - label_size[0] // 2
        
        # IMPORTANT FIX: Move labels further UP using configurable offset
        label_y = mouth_center_y - self.label_offset_y - max(label_size[1], threshold_size[1]) - label_padding
        
        # Ensure labels stay within image bounds
        label_y = max(label_size[1] + label_padding, label_y)
        
        # Draw background for status label
        cv2.rectangle(
            annotated_image,
            (label_x - label_padding, label_y - label_size[1] - label_padding),
            (label_x + label_size[0] + label_padding, label_y + label_padding),
            mouth_color, -1
        )
        
        # Draw status text
        cv2.putText(
            annotated_image, label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            adaptive_font_scale,
            (255, 255, 255),  # White text
            adaptive_font_thickness
        )
        
        # Draw threshold label below status
        threshold_x = mouth_center_x - threshold_size[0] // 2
        threshold_y = label_y + label_size[1] + label_padding * 2
        
        # Draw background for threshold label
        cv2.rectangle(
            annotated_image,
            (threshold_x - label_padding, threshold_y - threshold_size[1] - label_padding),
            (threshold_x + threshold_size[0] + label_padding, threshold_y + label_padding),
            (100, 100, 100), -1  # Gray background
        )
        
        # Draw threshold text
        cv2.putText(
            annotated_image, threshold_label,
            (threshold_x, threshold_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            adaptive_font_scale * 0.8,
            (255, 255, 255),  # White text
            adaptive_font_thickness
        )
    
    def _publish_speaking_visualization(
        self, 
        cv_image: np.ndarray, 
        landmarks: List[Tuple[float, float, float]], 
        is_speaking: bool,
        speaking_confidence: float,
        face_id: str,
        header: Header,
        mouth_crop_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Publish annotated image with lip visualization.
        
        Draws mouth landmarks with color coding for dlib mode, or
        draws mouth crop bbox and YOLO landmarks for YOLO mode.
        - Red: Not speaking (confidence < speaking_threshold)
        - Green: Speaking (confidence >= speaking_threshold)
        
        Also displays the speaking threshold on the image.
        
        Args:
            cv_image: Original OpenCV image
            landmarks: List of (x, y, c) landmark coordinates with confidence
            is_speaking: Speaking detection result
            speaking_confidence: Speaking confidence score
            face_id: Face identifier for labeling
            header: Original image header
            mouth_crop_bbox: Optional mouth crop bbox (x1, y1, x2, y2) for visualization
        """
        try:
            # Create a copy for annotation
            annotated_image = cv_image.copy()
            
            # Detect if we have YOLO or dlib landmarks
            valid_landmarks = sum(1 for lm in landmarks if len(lm) >= 3 and lm[2] > 0.0)
            is_yolo = valid_landmarks < 10
            
            # Calculate adaptive sizes based on image dimensions
            img_height, img_width = annotated_image.shape[:2]
            base_size = min(img_width, img_height)
            
            adaptive_line_thickness = max(2, int(base_size * 0.003))  # 0.3% of image size
            adaptive_landmark_radius = max(3, int(base_size * 0.008))  # 0.8% of image size
            adaptive_font_scale = max(0.6, base_size * 0.0012)  # Adaptive font size
            adaptive_font_thickness = max(2, int(base_size * 0.002))  # Adaptive font thickness
            
            # Choose color based on speaking status
            if is_speaking:
                mouth_color = (0, 255, 0)  # Green for speaking
                status_text = "SPEAKING"
            else:
                mouth_color = (0, 0, 255)  # Red for not speaking
                status_text = "NOT SPEAKING"
            
            # Visualization based on landmark type
            if is_yolo:
                # YOLO mode: Draw the crop rectangle and the 2 mouth landmarks
                if mouth_crop_bbox is not None:
                    x1, y1, x2, y2 = mouth_crop_bbox
                    # Draw thick rectangle for the mouth crop area
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), mouth_color, adaptive_line_thickness * 2)
                    # Add label on the rectangle
                    cv2.putText(annotated_image, "MOUTH CROP", (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale * 0.7, mouth_color, adaptive_font_thickness)
                
                # Draw YOLO mouth corner landmarks
                if len(landmarks) > 48 and landmarks[48][2] > 0:  # Left mouth
                    pt = (int(landmarks[48][0]), int(landmarks[48][1]))
                    cv2.circle(annotated_image, pt, adaptive_landmark_radius * 2, (255, 0, 0), -1)  # Blue
                    cv2.putText(annotated_image, "L", (pt[0] + 10, pt[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, (255, 0, 0), adaptive_font_thickness)
                
                if len(landmarks) > 54 and landmarks[54][2] > 0:  # Right mouth
                    pt = (int(landmarks[54][0]), int(landmarks[54][1]))
                    cv2.circle(annotated_image, pt, adaptive_landmark_radius * 2, (255, 0, 0), -1)  # Blue
                    cv2.putText(annotated_image, "R", (pt[0] + 10, pt[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, (255, 0, 0), adaptive_font_thickness)
                
                # Calculate label position from mouth crop bbox or landmarks
                if mouth_crop_bbox is not None:
                    mouth_center_x = (mouth_crop_bbox[0] + mouth_crop_bbox[2]) // 2
                    mouth_center_y = (mouth_crop_bbox[1] + mouth_crop_bbox[3]) // 2
                elif len(landmarks) > 54 and landmarks[48][2] > 0 and landmarks[54][2] > 0:
                    mouth_center_x = int((landmarks[48][0] + landmarks[54][0]) / 2)
                    mouth_center_y = int((landmarks[48][1] + landmarks[54][1]) / 2)
                else:
                    mouth_center_x = img_width // 2
                    mouth_center_y = img_height // 2
            else:
                # dlib mode: Draw full mouth contours
                if len(landmarks) < 68:
                    if self.enable_debug_output:
                        self.get_logger().warn(f"Expected 68 landmarks for dlib visualization, got {len(landmarks)}. Skipping.")
                    return
                
                # Draw mouth outer contour (landmarks 48-59)
                for i in range(48, 59):
                    pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
                    pt2 = (int(landmarks[i + 1][0]), int(landmarks[i + 1][1]))
                    cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
                
                # Close mouth outer contour
                pt1 = (int(landmarks[59][0]), int(landmarks[59][1]))
                pt2 = (int(landmarks[48][0]), int(landmarks[48][1]))
                cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
                
                # Draw mouth inner contour (landmarks 60-67)
                for i in range(60, 67):
                    pt1 = (int(landmarks[i][0]), int(landmarks[i][1]))
                    pt2 = (int(landmarks[i + 1][0]), int(landmarks[i + 1][1]))
                    cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
                
                # Close mouth inner contour
                pt1 = (int(landmarks[67][0]), int(landmarks[67][1]))
                pt2 = (int(landmarks[60][0]), int(landmarks[60][1]))
                cv2.line(annotated_image, pt1, pt2, mouth_color, adaptive_line_thickness)
                
                # Draw mouth landmarks as circles
                for i in range(48, 68):  # Mouth landmarks (48-67)
                    lm_x, lm_y = int(landmarks[i][0]), int(landmarks[i][1])
                    cv2.circle(annotated_image, (lm_x, lm_y), adaptive_landmark_radius, mouth_color, -1)
                
                # Calculate mouth center for label placement
                mouth_landmarks = landmarks[48:68]
                mouth_center_x = int(np.mean([lm[0] for lm in mouth_landmarks]))
                mouth_center_y = int(np.mean([lm[1] for lm in mouth_landmarks]))
            
            # Format speaking threshold to first 2 digits (e.g., 0.5 -> "0.5")
            threshold_str = f"{self.speaking_threshold:.1f}"
            
            # Create label with status, confidence, and threshold
            label = f"{status_text} ({speaking_confidence:.2f})"
            threshold_label = f"Threshold: {threshold_str}"
            
            # Draw label background and text near mouth
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                        adaptive_font_scale, adaptive_font_thickness)[0]
            threshold_size = cv2.getTextSize(threshold_label, cv2.FONT_HERSHEY_SIMPLEX,
                                            adaptive_font_scale * 0.8, adaptive_font_thickness)[0]
            
            label_padding = max(5, int(base_size * 0.01))
            label_x = mouth_center_x - label_size[0] // 2
            label_y = mouth_center_y - max(label_size[1], threshold_size[1]) - label_padding * 3
            
            # Draw background for status label
            cv2.rectangle(
                annotated_image,
                (label_x - label_padding, label_y - label_size[1] - label_padding),
                (label_x + label_size[0] + label_padding, label_y + label_padding),
                mouth_color, -1
            )
            
            # Draw status text
            cv2.putText(
                annotated_image, label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                adaptive_font_scale,
                (255, 255, 255),  # White text
                adaptive_font_thickness
            )
            
            # Draw threshold label below status
            threshold_x = mouth_center_x - threshold_size[0] // 2
            threshold_y = label_y + label_size[1] + label_padding * 2
            
            # Draw background for threshold label
            cv2.rectangle(
                annotated_image,
                (threshold_x - label_padding, threshold_y - threshold_size[1] - label_padding),
                (threshold_x + threshold_size[0] + label_padding, threshold_y + label_padding),
                (100, 100, 100), -1  # Gray background
            )
            
            # Draw threshold text
            cv2.putText(
                annotated_image, threshold_label,
                (threshold_x, threshold_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                adaptive_font_scale * 0.8,
                (255, 255, 255),  # White text
                adaptive_font_thickness
            )
            
            # Convert to ROS Image and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = header
            self.image_publisher.publish(annotated_msg)
            
            if self.enable_debug_output:
                self.get_logger().debug(
                    f"Published speaking visualization for {face_id}: "
                    f"{status_text} (confidence={speaking_confidence:.2f}, threshold={threshold_str})"
                )
        
        except Exception as e:
            self.get_logger().error(f"Error publishing speaking visualization: {e}")


def main(args=None):
    """Main entry point for the visual speech activity node."""
    rclpy.init(args=args)
    
    try:
        node = VisualSpeechActivityNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in visual speech activity node: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
