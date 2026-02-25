#!/usr/bin/env python3
"""
Face Recognition Node for HRI Applications

This node subscribes to FacialLandmarksArray messages from face detection,
extracts face embeddings, performs identity clustering and temporal tracking,
and publishes FacialRecognition messages following the ros4hri standard.

The approach is 100% based on the EUT YOLO identity management system,
providing persistent identity tracking across changing track IDs.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2
import time
import os
from typing import Dict, List, Optional, Tuple, Any

try:
    from hri_msgs.msg import FacialLandmarks, FacialLandmarksArray, FacialRecognition, FacialRecognitionArray, IdsList
except ImportError:
    print("Warning: hri_msgs not found. Please install hri_msgs package.")
    FacialLandmarks = None
    FacialLandmarksArray = None
    FacialRecognition = None
    FacialRecognitionArray = None

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from .face_embedding_extractor import create_face_embedding_extractor
from .identity_manager import IdentityManager


class FaceRecognitionNode(Node):
    """
    ROS2 node for face recognition using embedding-based identity management.
    
    Subscribes to FacialLandmarksArray messages and publishes FacialRecognition messages
    with persistent identity tracking based on face embeddings and clustering.
    """
    
    def __init__(self):
        super().__init__('face_recognition_node')
        
        # Declare parameters
        self._declare_parameters()
        
        # Initialize components
        self.cv_bridge = CvBridge()
        self.face_embedding_extractor = None
        self.identity_manager = None

        # Initialize image storage variables (copied from perception node)
        self.latest_color_image_msg = None
        self.color_image_processed = False
        self.latest_color_image_timestamp = None
        
        # Minimum height size for face detection (pixels) default 40 if no ros param
        self.min_h_size = self.get_parameter('min_h_size').get_parameter_value().integer_value
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.processed_messages = 0
        
        # Debug settings
        self.enable_debug_output = False  # Will be set during initialization

        # Setup QoS profiles (copied from perception node)
        self.qos_profile = QoSProfile(
            depth=1,  # Keep only the latest image
            # reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # durability=DurabilityPolicy.VOLATILE,
            # history=QoSHistoryPolicy.KEEP_LAST,
        )
    
        
        # Setup subscribers and publishers
        self._setup_topics()
        
        # Create RGB-only subscriber (copied from perception node RGB-only pattern)
        if self.enable_image_output:
            self.get_logger().debug("Setting up RGB-only processing for face recognition")
            
            # Create image subscribers - choose between compressed and regular image
            compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
            if compressed_topic and compressed_topic.strip():
                self.get_logger().info(f"Using compressed image topic: {compressed_topic}")
                real_time_qos = QoSProfile(
                    depth=1,  # Keep only latest image
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # No retransmissions
                    history=QoSHistoryPolicy.KEEP_LAST,
                    durability=DurabilityPolicy.VOLATILE  # Don't persist messages
                )
                self.color_sub = self.create_subscription(
                    CompressedImage, 
                    compressed_topic, 
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

        # Store latest landmarks for processing (array mode)
        self.latest_landmarks_array = None
        self.landmarks_processed = False
        
        # Message buffer for ROS4HRI with ID mode (sync by timestamp)
        # Will be initialized in _setup_topics after parameter is available
        self.landmarks_buffer = {}  # {timestamp: [FacialLandmarks, ...]}
        self.buffer_timeout = 0.1  # 100ms timeout for frame synchronization
        self.last_processed_timestamp = None
        
        # Frame skipping for recognition optimization
        # Cache of last recognition results per face_id: {face_id: (unique_id, confidence, timestamp)}
        self.recognition_cache = {}
        # Global frame counter for frame-level skipping (not per-face)
        self.global_frame_counter = 0

        # Timer for periodic inference (copied from perception node pattern)
        timer_period = 1.0 / self.processing_rate_hz  # Use processing_rate_hz parameter
        self.inference_timer = self.create_timer(
            timer_period, 
            self.inference_timer_callback
        )
        
        # Initialize face embedding extractor and identity manager
        self._initialize_components()
        
        # Timer for processing buffered frames in ROS4HRI with ID mode (after _setup_topics sets ros4hri_with_id)
        # Note: ros4hri_with_id is set in _setup_topics which is called earlier
        if hasattr(self, 'ros4hri_with_id') and self.ros4hri_with_id:
            buffer_timer_period = 0.05  # Check buffer every 50ms
            self.buffer_timer = self.create_timer(
                buffer_timer_period,
                self.process_buffered_frames
            )
        
        self.get_logger().debug("Face Recognition Node initialized")
        self.get_logger().debug(f"Processing rate: {self.processing_rate_hz} Hz")

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
        # self.get_logger().debug("Color image received.")

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
        self.get_logger().debug("Compressed color image received.")

    # -------------------------------------------------------------------------
    #                         Timer Callback for Inference
    # -------------------------------------------------------------------------
    def inference_timer_callback(self):
        """
        Regular callback for continuous inference mode.

        Triggered by the timer at the configured frequency.
        Processes latest landmarks and image data for face recognition.
        """
        
        start_time = self.get_clock().now()

        # Check if we have new landmarks to process
        if self.latest_landmarks_array is None:
            # self.get_logger().warning("No landmarks data received")
            return
        if self.landmarks_processed is True:
            return

        # If image processing is enabled, check for image data
        color_msg = self.latest_color_image_msg
        color_image_processed = self.color_image_processed
        
        if color_msg is None:
            # self.get_logger().warning("No image data received for face recognition")
            return
        if color_image_processed is True:
            return
        
        # Convert image to OpenCV format
        try:
            compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
            if compressed_topic and compressed_topic.strip():
                # Handle compressed image
                np_arr = np.frombuffer(color_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if cv_image is None:
                    self.get_logger().error('Failed to decode compressed image')
                    return
            else:
                # Handle regular image
                cv_image = self.cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
        
        self.color_image_processed = True
        
        if cv_image is None or cv_image.size == 0:
            self.get_logger().warn("Received empty or invalid image")
            return
            
        # Store image for processing
        self.last_image = cv_image
        self.last_image_header = color_msg.header


        # Mark landmarks as processed
        landmarks_msg = self.latest_landmarks_array
        self.landmarks_processed = True

        try:
            # Process the landmarks array
            self.process_landmarks_array(landmarks_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error processing face recognition: {e}")

        # Calculate and log timing information
        end_time = self.get_clock().now()
        processing_time = (end_time - start_time).nanoseconds / 1e6  # Convert to milliseconds
        
        # Update timing statistics
        self.processed_messages += 1
        self.total_processing_time += processing_time
        
        if not hasattr(self, 'max_processing_time'):
            self.max_processing_time = processing_time
            self.min_processing_time = processing_time
        else:
            self.max_processing_time = max(self.max_processing_time, processing_time)
            self.min_processing_time = min(self.min_processing_time, processing_time)
        
        # Log timing every 100 frames or when debug is enabled
        if self.processed_messages % 100 == 0:
            avg_time = self.total_processing_time / self.processed_messages
            faces_count = len(landmarks_msg.ids) if landmarks_msg and landmarks_msg.ids else 0
            self.get_logger().info(
                f"[TIMING] Face Recognition - Frame #{self.processed_messages}: "
                f"Current: {processing_time:.2f}ms, "
                f"Avg: {avg_time:.2f}ms, "
                f"Min: {self.min_processing_time:.2f}ms, "
                f"Max: {self.max_processing_time:.2f}ms, "
                f"Faces: {faces_count}"
            )

    def process_landmarks_array(self, msg):
        """
        Process landmarks array and perform face recognition for all faces.
        
        Args:
            msg: FacialLandmarksArray message
        """
        if not self.face_embedding_extractor or not self.identity_manager:
            self.get_logger().warning("Face recognition components not initialized")
            # Still publish image even if components not initialized
            if self.enable_image_output and self.image_output_publisher and self.last_image is not None:
                self._publish_clean_image()
            return
        
        if self.last_image is None:
            self.get_logger().warning("No image available for face recognition")
            return
            
        # Handle empty messages
        if not msg.ids:
            if self.enable_debug_output:
                self.get_logger().debug('Received empty FacialLandmarksArray - publishing clean image')
            # Publish clean image when no faces detected
            if self.enable_image_output and self.image_output_publisher:
                self._publish_clean_image()
            # In ROS4HRI with ID mode, we don't publish recognition messages for empty frames
            if not self.ros4hri_with_id:
                self._publish_recognition_array([])
            return
        
        if self.enable_debug_output:
            self.get_logger().debug(f'Processing FacialLandmarksArray with {len(msg.ids)} faces')
        
        try:
            # Process all faces in batch mode
            self._process_landmarks_array_batch(msg)
        
        except Exception as e:
            self.get_logger().error(f"Error in landmarks array processing: {e}")
            # Still publish clean image on error
            if self.enable_image_output and self.image_output_publisher:
                self._publish_clean_image()
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Input/Output topics
        self.declare_parameter('compressed_topic', '')
        self.declare_parameter('input_topic', '/humans/faces/detected')
        self.declare_parameter('output_topic', '/humans/faces/recognized')
        self.declare_parameter('image_input_topic', '/camera/color/image_rect_raw')
        
        # Processing rate parameter (copied from perception node)
        self.declare_parameter('processing_rate_hz', 10.0)  # Default 10 Hz
        self.declare_parameter('min_h_size', 40)  # Minimum height size for valid face detection (pixels)

        # Get processing rate parameter (copied from perception node)
        self.processing_rate_hz = self.get_parameter('processing_rate_hz').get_parameter_value().double_value
        
        # Image output parameters
        self.declare_parameter('enable_image_output', True)
        self.declare_parameter('img_published_reshape_size', [640, 360])  # Resolution for published annotated images
        self.declare_parameter('output_image_topic', '/humans/faces/recognized/annotated_img/compressed')
        
        # Face embedding parameters
        self.declare_parameter('face_embedding_model', 'vggface2')
        self.declare_parameter('device', 'cpu')  # or 'cuda'
        self.declare_parameter('weights_path', 'weights')
        self.declare_parameter('face_embedding_weights_name', '')
        
        # Identity management parameters
        self.declare_parameter('max_embeddings_per_identity', 50)
        self.declare_parameter('similarity_threshold', 0.6)
        self.declare_parameter('track_identity_stickiness_margin', 0.4) 
        self.declare_parameter('clustering_threshold', 0.7)
        self.declare_parameter('embedding_inclusion_threshold', 0.6)
        self.declare_parameter('identity_timeout', 60.0)
        self.declare_parameter('min_detections_for_stable_identity', 5)
        self.declare_parameter('enable_debug_output', True)  # Temporarily enable for debugging
        self.declare_parameter('use_ewma_for_mean', False)
        self.declare_parameter('ewma_alpha', 0.6)
        self.declare_parameter('min_embeddings_for_identity', 5)  # Minimum embeddings required to consider an identity valid (used for cleanup of inactive identities)
        
        # MongoDB parameters for identity storage
        self.declare_parameter('use_mongodb', True)  # Whether to use MongoDB for identity persistence
        self.declare_parameter('mongo_uri', 'mongodb://eurecat:cerdanyola@localhost:27018/?authSource=admin&serverSelectionTimeoutMS=5000') #'mongodb://localhost:27018/')# #eurecat:cerdanyola@mongodb:27018/
        self.declare_parameter('mongo_db_name', 'face_recognition_db')
        self.declare_parameter('mongo_collection_name', 'identity_database')
        
        # Processing parameters
        self.declare_parameter('gaze_identity_exclusion_threshold', 0.5)
        
        # Frame skipping optimization parameter
        # Percentage of frames to skip recognition processing (0-100)
        # When skipping, the last cached recognition result for that face_id will be reused
        self.declare_parameter('recognition_frame_skip_percentage', 0.0)  # Default: no skipping
        
        # Receiver ID for hri_msgs
        self.declare_parameter('receiver_id', 'face_recognition')
        
        # ROS4HRI mode parameter - when enabled, subscribes to per-ID messages and publishes per-ID
        self.declare_parameter('ros4hri_with_id', False)  # Default to array mode (ROS4HRI array)
    
    def _setup_topics(self):
        """Setup ROS2 subscribers and publishers."""
        # Get topic names from parameters
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.image_input_topic = self.get_parameter('image_input_topic').get_parameter_value().string_value
        
        # Get ROS4HRI mode parameter
        self.ros4hri_with_id = self.get_parameter('ros4hri_with_id').get_parameter_value().bool_value
        
        if self.ros4hri_with_id:
            # ROS4HRI with ID mode: Subscribe to tracked faces list and per-ID topics
            # Dictionary to store subscribers for each face ID: {face_id: subscriber}
            self.landmarks_subscribers = {}  # {face_id: Subscription}
            self.recognition_publishers = {}  # {face_id: Publisher}
            self.tracked_face_ids = set()  # Set of currently tracked face IDs
            
            # Subscribe to tracked faces list
            self.tracked_faces_subscriber = self.create_subscription(
                IdsList,
                '/humans/faces/tracked',
                self.tracked_faces_callback,
                self.qos_profile
            )
            self.get_logger().info("ROS4HRI with ID mode enabled: Subscribing to /humans/faces/tracked and per-ID topics")
            self.landmarks_subscriber = None
        else:
            # ROS4HRI array mode: Subscribe to FacialLandmarksArray messages
            self.landmarks_subscriber = self.create_subscription(
                FacialLandmarksArray,
                input_topic,
                self.landmarks_array_callback,
                self.qos_profile
            )
            self.get_logger().info("ROS4HRI array mode enabled: Subscribing to FacialLandmarksArray messages")
            self.landmarks_subscribers = {}
            self.recognition_publishers = {}
            self.tracked_face_ids = set()
            self.tracked_faces_subscriber = None
        
        # Create publisher based on mode - only one publisher per topic
        if self.ros4hri_with_id:
            # ROS4HRI with ID mode: Publishers will be created dynamically per face ID
            # (recognition_publishers dictionary is already initialized above)
            self.recognition_publisher = None
        else:
            # ROS4HRI array mode: Publish FacialRecognitionArray
            self.recognition_publisher = self.create_publisher(
                FacialRecognitionArray,
                output_topic,
                self.qos_profile
            )
        
        # Image output publisher (optional)
        self.enable_image_output = self.get_parameter('enable_image_output').get_parameter_value().bool_value
        self.img_published_reshape_size = self.get_parameter('img_published_reshape_size').get_parameter_value().integer_array_value
        if self.enable_image_output:
            output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
            image_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,   # 1–5 is ideal for images over Wi-Fi
                durability=DurabilityPolicy.VOLATILE
            )
            self.image_output_publisher = self.create_publisher(
                CompressedImage,
                output_image_topic,
                image_qos
            )
            self.get_logger().debug(f"Compressed image output enabled: {output_image_topic}")
        else:
            self.image_output_publisher = None
        
        self.get_logger().debug(f"Subscribed to: {input_topic}")
        self.get_logger().debug(f"Publishing to: {output_topic}")
        if self.ros4hri_with_id:
            self.get_logger().debug("Publishing individual FacialRecognition messages")
        else:
            self.get_logger().debug("Publishing FacialRecognitionArray messages")
    
    def _initialize_components(self):
        """Initialize face embedding extractor and identity manager."""
        # Get parameters
        face_embedding_model = self.get_parameter('face_embedding_model').get_parameter_value().string_value
        device_param = self.get_parameter('device').get_parameter_value().string_value
        
        # Handle device parameter: if cuda is requested but not available, fall back to cpu
        device = device_param
        if 'cuda' in device_param.lower():
            try:
                import torch
                if not torch.cuda.is_available():
                    self.get_logger().warn(f"CUDA requested but not available. Falling back to CPU.")
                    device = 'cpu'
            except ImportError:
                self.get_logger().warn(f"PyTorch not available. Using CPU.")
                device = 'cpu'
        
        face_embedding_weights_name = self.get_parameter('face_embedding_weights_name').get_parameter_value().string_value
        
        # Build full weights path similar to YOLO approach
        weights_dir_path = None
        face_embedding_weights_path = None
        
        # Get the current file's directory and navigate to package root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_src_dir = os.path.dirname(current_file_dir)
        package_src_dir = package_src_dir.replace('build', 'src') #save it in docker volume of ros2 package
        weights_dir_path = os.path.join(package_src_dir,"weights")
        #create weights dir if not there
        if not os.path.exists(weights_dir_path):
            os.makedirs(weights_dir_path) 
        # If specific weights filename is provided, build full path
        if not face_embedding_weights_name:
            self.get_logger().error("No specific face embedding weights name provided; using default vggface2 for model")
            face_embedding_weights_name = '20180402-114759-vggface2.pt'  # Default for vggface2 model

        face_embedding_weights_path = os.path.join(weights_dir_path, face_embedding_weights_name)
        self.get_logger().debug(f"Face embedding weights path: {face_embedding_weights_path}")
        # Initialize face embedding extractor
        try:
            self.face_embedding_extractor = create_face_embedding_extractor(
                model_name=face_embedding_model,
                device=device,
                weights_path=weights_dir_path,
                face_embedding_weights_path=face_embedding_weights_path
            )
            
            if self.face_embedding_extractor.is_available():
                self.get_logger().debug(f"Face embedding extractor initialized: {face_embedding_model} on {device}")
                model_info = self.face_embedding_extractor.get_model_info()
                self.get_logger().debug(f"Model debug: {model_info}")
            else:
                self.get_logger().error("Face embedding extractor failed to initialize")
                return
                
        except Exception as e:
            self.get_logger().error(f"Failed to initialize face embedding extractor: {e}")
            return
        
        # Initialize identity manager
        try:
            max_embeddings = self.get_parameter('max_embeddings_per_identity').get_parameter_value().integer_value
            similarity_thresh = self.get_parameter('similarity_threshold').get_parameter_value().double_value
            stickiness_margin = self.get_parameter('track_identity_stickiness_margin').get_parameter_value().double_value
            clustering_thresh = self.get_parameter('clustering_threshold').get_parameter_value().double_value
            embedding_inclusion_thresh = self.get_parameter('embedding_inclusion_threshold').get_parameter_value().double_value
            identity_timeout = self.get_parameter('identity_timeout').get_parameter_value().double_value
            min_detections = self.get_parameter('min_detections_for_stable_identity').get_parameter_value().integer_value
            debug_prints = self.get_parameter('enable_debug_output').get_parameter_value().bool_value
            use_ewma = self.get_parameter('use_ewma_for_mean').get_parameter_value().bool_value
            ewma_alpha = self.get_parameter('ewma_alpha').get_parameter_value().double_value
            min_emin_embeddings_for_identity = self.get_parameter('min_embeddings_for_identity').get_parameter_value().integer_value

            # MongoDB parameters
            use_mongodb = self.get_parameter('use_mongodb').get_parameter_value().bool_value    
            mongo_uri = self.get_parameter('mongo_uri').get_parameter_value().string_value
            mongo_db_name = self.get_parameter('mongo_db_name').get_parameter_value().string_value
            mongo_collection_name = self.get_parameter('mongo_collection_name').get_parameter_value().string_value
            
            # Set debug prints flag
            self.enable_debug_output = debug_prints
            
            self.identity_manager = IdentityManager(
                logger=self.get_logger(),
                max_embeddings_per_identity=max_embeddings,
                similarity_threshold=similarity_thresh,
                track_identity_stickiness_margin=stickiness_margin,
                clustering_threshold=clustering_thresh,
                embedding_inclusion_threshold=embedding_inclusion_thresh,
                identity_timeout=identity_timeout,
                min_detections_for_stable_identity=min_detections,
                enable_debug_output=debug_prints,
                mongo_uri=mongo_uri,
                mongo_db_name=mongo_db_name,
                mongo_collection_name=mongo_collection_name,
                use_ewma_for_mean=use_ewma,
                ewma_alpha=ewma_alpha,
                use_mongodb=use_mongodb,
                min_embeddings_for_identity=min_emin_embeddings_for_identity
            )
            
            self.get_logger().info("Identity manager initialized")
            self.get_logger().info(f"Parameters: similarity_threshold={similarity_thresh}, clustering_threshold={clustering_thresh}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize identity manager: {e}")
            return
    
    def landmarks_array_callback(self, msg):
        """
        Callback for processing array of facial landmarks and computing face recognition for all faces.
        
        Args:
            msg: FacialLandmarksArray message containing multiple face landmarks
        """
        # Store latest landmarks for processing
        self.latest_landmarks_array = msg
        self.landmarks_processed = False
        # self.get_logger().info("landmarks arrive")
    
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
                    lambda m, fid=face_id: self.landmarks_individual_callback(m, fid),
                    self.qos_profile
                )
                
                # Create publisher for this face ID
                output_topic_name = f'/humans/faces/{face_id}/recognized'
                self.recognition_publishers[face_id] = self.create_publisher(
                    FacialRecognition,
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
    
    def landmarks_individual_callback(self, msg, face_id):
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
    
    def _should_skip_recognition(self):
        """
        Determine if recognition should be skipped for this entire frame based on frame skip percentage.
        This operates at the FRAME level, not per-face - if a frame is skipped, ALL faces in that frame are skipped.
        
        Returns:
            bool: True if this entire frame should be skipped, False otherwise
        """
        skip_percentage = self.get_parameter('recognition_frame_skip_percentage').get_parameter_value().double_value
        
        # If skip percentage is 0, never skip
        if skip_percentage <= 0.0:
            return False
        
        # If skip percentage is 100, always skip (but still process first frame)
        if skip_percentage >= 100.0:
            return self.global_frame_counter > 0
        
        # Calculate skip pattern: if skip_percentage is 50%, process every other frame (skip 1, process 1)
        # frames_per_cycle = 100 / skip_percentage rounded
        # For 50%: every 2 frames, skip 1, process 1
        # For 75%: every 4 frames, skip 3, process 1
        
        frames_per_cycle = max(1, int(100.0 / skip_percentage))
        frames_to_skip = max(1, int(skip_percentage / 100.0 * frames_per_cycle))
        
        frame_position = self.global_frame_counter % frames_per_cycle
        
        # Skip if we're in the skip portion of the cycle
        should_skip = frame_position < frames_to_skip
        
        if self.enable_debug_output and should_skip:
            self.get_logger().debug(f"Skipping entire frame {self.global_frame_counter} (position {frame_position}/{frames_per_cycle}, skip_pct={skip_percentage}%)")
        
        return should_skip
    
    def _process_landmarks_array_batch(self, msg):
        """Process array of facial landmarks in batch mode for better performance."""
        if not msg.ids:
            return
        
        # Increment global frame counter
        self.global_frame_counter += 1
        
        # Check if we should skip this ENTIRE frame
        if self._should_skip_recognition():
            # Skip the entire frame - publish cached results for all faces
            if self.enable_debug_output:
                self.get_logger().info(f"Skipping frame {self.global_frame_counter} - publishing {len(msg.ids)} cached recognitions")
            
            all_recognition_results = []
            for facial_landmarks_msg in msg.ids:
                face_id = facial_landmarks_msg.face_id
                
                # Try to get cached result
                if face_id in self.recognition_cache:
                    unique_id, confidence, _ = self.recognition_cache[face_id]
                    # Format as tuple: (landmarks_msg, unique_id, confidence)
                    all_recognition_results.append((facial_landmarks_msg, unique_id, confidence))
                else:
                    # No cached result - publish unknown
                    if self.enable_debug_output:
                        self.get_logger().debug(f"No cached recognition for {face_id}, publishing unknown")
                    # Format as tuple: (landmarks_msg, unique_id, confidence)
                    all_recognition_results.append((facial_landmarks_msg, "unknown", 0.0))
            
            # Publish all cached/unknown results based on mode
            if self.ros4hri_with_id:
                # ROS4HRI with ID mode: Publish to per-ID topics
                for landmarks_msg, unique_id, confidence in all_recognition_results:
                    face_id = landmarks_msg.face_id
                    
                    # Create publisher for this face ID if it doesn't exist
                    if face_id not in self.recognition_publishers:
                        topic_name = f'/humans/faces/{face_id}/recognized'
                        self.recognition_publishers[face_id] = self.create_publisher(
                            FacialRecognition,
                            topic_name,
                            self.qos_profile
                        )
                    
                    recognition_msg = FacialRecognition()
                    recognition_msg.header = landmarks_msg.header
                    recognition_msg.face_id = face_id
                    recognition_msg.recognized_face_id = unique_id if unique_id != "unknown" else "unknown"
                    recognition_msg.confidence = float(confidence)
                    
                    # Initialize speaking fields if they exist
                    if hasattr(recognition_msg, 'is_speaking'):
                        recognition_msg.is_speaking = False
                        recognition_msg.speaking_confidence = 0.0
                    
                    # Publish to the per-ID topic
                    self.recognition_publishers[face_id].publish(recognition_msg)
            else:
                # ROS4HRI array mode: Publish FacialRecognitionArray
                self._publish_recognition_array(all_recognition_results)
            return
        
        # Process this frame normally (don't skip)
        if self.enable_debug_output:
            self.get_logger().info(f"Processing frame {self.global_frame_counter} with {len(msg.ids)} faces")
        
        # Extract face crops and face IDs for all faces
        if self.enable_debug_output:
            crop_start_time = time.time()
        
        face_crops = []
        face_ids = []
        landmarks_msgs = []
        
        for facial_landmarks_msg in msg.ids:
            face_crop = self._extract_face_crop_from_landmarks(facial_landmarks_msg)
            if face_crop is not None:
                face_crops.append(face_crop)
                face_ids.append(facial_landmarks_msg.face_id)
                landmarks_msgs.append(facial_landmarks_msg)
        
        if self.enable_debug_output:
            crop_time = (time.time() - crop_start_time) * 1000
            self.get_logger().info(f"Face crop extraction took: {crop_time:.2f}ms for {len(msg.ids)} input faces, got {len(face_crops)} valid crops")
        
        # Collect all recognition results
        all_recognition_results = []
        
        # Process faces that need recognition
        if face_crops:
            # Check if face embedding extractor is available
            if not self.face_embedding_extractor.is_available():
                self.get_logger().error("Face embedding extractor is not available")
                return
            
            # Extract embeddings in batch
            if self.enable_debug_output:
                embedding_start_time = time.time()
            
            embeddings = self.face_embedding_extractor.extract_embeddings_batch(face_crops)
            
            if self.enable_debug_output:
                embedding_time = (time.time() - embedding_start_time) * 1000
                self.get_logger().debug(f"Embedding extraction took: {embedding_time:.2f}ms for {len(face_crops)} faces")
            
            # Create face_embeddings dictionary for identity manager using face_id as key
            if self.enable_debug_output:
                prep_start_time = time.time()
            
            face_embeddings = {}
            valid_indices = []
            
            for i, (face_id, embedding) in enumerate(zip(face_ids, embeddings)):
                if embedding is not None:
                    face_embeddings[face_id] = embedding
                    valid_indices.append(i)
                else:
                    self.get_logger().warning(f"Failed to extract embedding for face {face_ids[i]}")
            
            if self.enable_debug_output:
                prep_time = (time.time() - prep_start_time) * 1000
                self.get_logger().debug(f"Embedding preparation took: {prep_time:.2f}ms")
            
            # Process identities in batch
            if face_embeddings:
                if self.enable_debug_output:
                    identity_start_time = time.time()
                
                identity_results = self.identity_manager.process_new_embedding_batch(face_embeddings)
                
                if self.enable_debug_output:
                    identity_time = (time.time() - identity_start_time) * 1000
                    self.get_logger().debug(f"Identity processing took: {identity_time:.2f}ms for {len(face_embeddings)} faces")
                
                # Collect results for batch publishing and update cache
                for i in valid_indices:
                    face_id = face_ids[i]
                    landmarks_msg = landmarks_msgs[i]
                    unique_id, confidence = identity_results.get(face_id, (None, 0.0))
                    
                    # Update recognition cache with current timestamp
                    current_time = self.get_clock().now()
                    self.recognition_cache[face_id] = (unique_id, confidence, current_time)
                    
                    if self.enable_debug_output:
                        self.get_logger().debug(f"Face {face_id} -> Identity: {unique_id}, Confidence: {confidence:.3f}")
                    
                    all_recognition_results.append((landmarks_msg, unique_id, confidence))
        
        # Publish recognition results based on mode
        if all_recognition_results:
            if self.enable_debug_output:
                publish_start_time = time.time()
            
            if self.ros4hri_with_id:
                # ROS4HRI with ID mode: Publish to per-ID topics /humans/faces/<faceID>/recognized
                # All messages from the same frame share the same timestamp for synchronization
                frame_timestamp = all_recognition_results[0][0].header.stamp if all_recognition_results else None
                for landmarks_msg, unique_id, confidence in all_recognition_results:
                    face_id = landmarks_msg.face_id
                    
                    # Create publisher for this face ID if it doesn't exist
                    if face_id not in self.recognition_publishers:
                        topic_name = f'/humans/faces/{face_id}/recognized'
                        self.recognition_publishers[face_id] = self.create_publisher(
                            FacialRecognition,
                            topic_name,
                            self.qos_profile
                        )
                        if self.enable_debug_output:
                            self.get_logger().debug(f"Created publisher for face ID: {topic_name}")
                    
                    recognition_msg = FacialRecognition()
                    recognition_msg.header = landmarks_msg.header
                    # Ensure all messages from the same frame have the same timestamp
                    if frame_timestamp:
                        recognition_msg.header.stamp = frame_timestamp
                    recognition_msg.face_id = face_id
                    if unique_id is not None:
                        recognition_msg.recognized_face_id = unique_id
                        recognition_msg.confidence = float(confidence)
                    else:
                        recognition_msg.recognized_face_id = "unknown"
                        recognition_msg.confidence = 0.0
                    
                    # Initialize speaking fields if they exist (will be updated by visual_speech_activity node)
                    if hasattr(recognition_msg, 'is_speaking'):
                        recognition_msg.is_speaking = False
                        recognition_msg.speaking_confidence = 0.0
                    
                    # Publish to the per-ID topic
                    self.recognition_publishers[face_id].publish(recognition_msg)
                    if self.enable_debug_output:
                        self.get_logger().debug(f"Published FacialRecognition for face_id={face_id} to /humans/faces/{face_id}/recognized, recognized_id={unique_id}")
            else:
                # ROS4HRI array mode: Publish FacialRecognitionArray
                self._publish_recognition_array(all_recognition_results)
            
            # Publish annotated image with all recognitions if enabled
            if self.enable_image_output and self.image_output_publisher and all_recognition_results:
                # Create image recognition results from all results (processed + cached)
                image_recognition_results = [(landmarks_msg, (unique_id, confidence)) for landmarks_msg, unique_id, confidence in all_recognition_results]
                self._publish_batch_annotated_image(image_recognition_results)
            
            if self.enable_debug_output:
                publish_time = (time.time() - publish_start_time) * 1000
                self.get_logger().debug(f"Publishing results took: {publish_time:.2f}ms for {len(all_recognition_results)} faces")
        else:
            # No valid faces to process
            if not face_crops:
                self.get_logger().warning("No valid face crops extracted from landmarks array")
                # Publish empty results when no valid faces
                if not self.ros4hri_with_id:
                    self._publish_recognition_array([])
                self._publish_batch_annotated_image(None)
    
    def _publish_recognition_array(self, recognition_results: List):
        """Publish facial recognition results as a single array message."""
        # Safety check: this method should only be called in array mode
        if self.ros4hri_with_id:
            self.get_logger().warning("_publish_recognition_array called in ros4hri_with_id mode - skipping")
            return
        
        # Additional safety check: ensure publisher exists
        if self.recognition_publisher is None:
            self.get_logger().error("Recognition publisher is None - cannot publish array")
            return
            
        try:
            # Create FacialRecognitionArray message
            recognition_array_msg = FacialRecognitionArray()
            
            # Set header with current timestamp if we have results, otherwise use current time
            if recognition_results:
                # Use header from first landmarks message
                recognition_array_msg.header = recognition_results[0][0].header
            else:
                recognition_array_msg.header = Header()
                recognition_array_msg.header.stamp = self.get_clock().now().to_msg()
                recognition_array_msg.header.frame_id = "camera_color_optical_frame"  # Default frame
            
            # Create individual recognition messages
            facial_recognition_msgs = []
            for landmarks_msg, unique_id, confidence in recognition_results:
                recognition_msg = FacialRecognition()
                
                # Copy header from landmarks message
                recognition_msg.header = landmarks_msg.header
                
                # Set face_id from original message
                recognition_msg.face_id = landmarks_msg.face_id
                
                # Set recognized face ID and confidence
                if unique_id is not None:
                    recognition_msg.recognized_face_id = unique_id
                    recognition_msg.confidence = float(confidence)
                else:
                    recognition_msg.recognized_face_id = "unknown"
                    recognition_msg.confidence = 0.0
                
                # Initialize speaking fields if they exist (will be updated by visual_speech_activity node)
                if hasattr(recognition_msg, 'is_speaking'):
                    recognition_msg.is_speaking = False
                    recognition_msg.speaking_confidence = 0.0
                
                facial_recognition_msgs.append(recognition_msg)
            
            # Set the array
            recognition_array_msg.facial_recognition = facial_recognition_msgs
            
            # Publish the array message
            self.recognition_publisher.publish(recognition_array_msg)
            
            if self.enable_debug_output:
                self.get_logger().debug(f"Published FacialRecognitionArray with {len(facial_recognition_msgs)} faces")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish recognition array: {e}")
    
    def _extract_face_crop_from_landmarks(self, msg) -> Optional[np.ndarray]:
        """Extract face crop from landmarks message using bounding box."""
        try:
            if self.enable_debug_output:
                self.get_logger().debug(f"Extracting face crop for face_id: {msg.face_id}")
                self.get_logger().debug(f"Bbox confidence: {msg.bbox_confidence}")
                self.get_logger().debug(f"Bbox xyxy type: {type(msg.bbox_xyxy)}")
                
            # Use face bounding box from bbox_xyxy (now NormalizedRegionOfInterest2D)
            if hasattr(msg.bbox_xyxy, 'xmin') and msg.bbox_confidence > 0:
                # bbox_xyxy is now NormalizedRegionOfInterest2D with normalized coordinates [0,1]
                # Denormalize to pixel coordinates
                x1_norm, y1_norm = msg.bbox_xyxy.xmin, msg.bbox_xyxy.ymin
                x2_norm, y2_norm = msg.bbox_xyxy.xmax, msg.bbox_xyxy.ymax
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int(x1_norm * msg.width)
                y1 = int(y1_norm * msg.height)
                x2 = int(x2_norm * msg.width)
                y2 = int(y2_norm * msg.height)
                
                # Convert to x, y, w, h format
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                #if wh is less than min_h_size pixels, consider it invalid and skip to landmark-based cropping
                # or if the box is wider than it is tall (which is unlikely for a face), also consider it invalid and skip to landmark-based cropping
                if h < self.min_h_size or w > h:
                    if self.enable_debug_output:
                        self.get_logger().warning(f"Bounding box too small or too wide (w={w}, h={h}), skipping to face id {msg.face_id}")
                    return None

                if self.enable_debug_output:
                    self.get_logger().debug(f"Normalized bbox: ({x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f})")
                    self.get_logger().debug(f"Pixel bbox: x={x}, y={y}, w={w}, h={h}")
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, self.last_image.shape[1] - x)
                h = min(h, self.last_image.shape[0] - y)
                
                if w > 0 and h > 0:
                    face_crop = self.last_image[y:y+h, x:x+w]
                    
                    # Apply histogram equalization for color balancing while maintaining compatibility
                    if face_crop.size > 0:
                        # Save original face crop for debugging
                        if self.enable_debug_output:
                            debug_filename_original = f"/workspace/src/face_recognition/weights/imgs/face_crop_original_{msg.face_id}.jpg"
                            # cv2.imwrite(debug_filename_original, face_crop)
                            # self.get_logger().debug(f"Saved original face crop: {debug_filename_original}")
                        
                        # # NOT PERFORMING CORRECTLY - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better lighting robustness
                        # # Convert BGR to LAB color space for better color preservation
                        # lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
                        # l_channel, a_channel, b_channel = cv2.split(lab)
                        
                        # # Apply CLAHE to the L (lightness) channel only to preserve color information
                        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        # l_channel_clahe = clahe.apply(l_channel)
                        
                        # # Merge channels back and convert to BGR
                        # lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
                        # face_crop_balanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
                        face_crop_balanced = face_crop  # Skip processing for now to maintain compatibility and avoid issues

                        # Save processed face crop for debugging
                        if self.enable_debug_output:
                            debug_filename_processed = f"/workspace/src/face_recognition/weights/imgs/face_crop_processed_{msg.face_id}.jpg"
                            # cv2.imwrite(debug_filename_processed, face_crop_balanced)
                            # self.get_logger().debug(f"Saved processed face crop: {debug_filename_processed}")
                        
                        if self.enable_debug_output:
                            self.get_logger().debug(f"Extracted and processed face crop from normalized bbox: {face_crop_balanced.shape}")
                        
                        return face_crop_balanced
                    else:
                        if self.enable_debug_output:
                            self.get_logger().debug("Face crop is empty")
                        return face_crop
                else:
                    if self.enable_debug_output:
                        self.get_logger().debug(f"Invalid bbox dimensions after bounds check: w={w}, h={h}")
            
            # Fallback: estimate bounding box from landmarks
            landmarks = []
            valid_landmarks = 0
            for landmark in msg.landmarks:
                if landmark.c > 0:  # Valid landmark confidence
                    # Convert normalized coordinates to pixel coordinates
                    x_pixel = int(landmark.x * msg.width)
                    y_pixel = int(landmark.y * msg.height)
                    landmarks.append([x_pixel, y_pixel])
                    valid_landmarks += 1
            
            if self.enable_debug_output:
                self.get_logger().debug(f"Valid landmarks found: {valid_landmarks}")
            
            if len(landmarks) >= 4:  # Need at least 4 landmarks
                landmarks = np.array(landmarks)
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"Landmark bounds: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                
                # Add margin around landmarks
                margin = 0.2
                width = x_max - x_min
                height = y_max - y_min
                x_margin = int(width * margin)
                y_margin = int(height * margin)
                
                x = max(0, int(x_min - x_margin))
                y = max(0, int(y_min - y_margin))
                w = min(int(width + 2 * x_margin), self.last_image.shape[1] - x)
                h = min(int(height + 2 * y_margin), self.last_image.shape[0] - y)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"Landmark-based crop: x={x}, y={y}, w={w}, h={h}")
                
                #if wh is less than min_h_size pixels, consider it invalid and skip to landmark-based cropping
                #or if the box is wider than it is tall (which is unlikely for a face), also consider it invalid and skip to landmark-based cropping
                if h < self.min_h_size or w > h:
                    if self.enable_debug_output:
                        self.get_logger().warning(f"Bounding box too small or too wide (w={w}, h={h}), skipping to face id {msg.face_id}")
                    return None

                if w > 0 and h > 0:
                    face_crop = self.last_image[y:y+h, x:x+w]
                    
                    # Apply histogram equalization for color balancing while maintaining compatibility
                    if face_crop.size > 0:
                        # # NOT PERFORMING CORRECTLY - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better lighting robustness
                        # # Convert BGR to LAB color space for better color preservation
                        # lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
                        # l_channel, a_channel, b_channel = cv2.split(lab)
                        
                        # # Apply CLAHE to the L (lightness) channel only to preserve color information
                        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        # l_channel_clahe = clahe.apply(l_channel)
                        
                        # # Merge channels back and convert to BGR
                        # lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
                        # face_crop_balanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
                        face_crop_balanced = face_crop  # Skip processing for now to maintain compatibility and avoid issues

                        if self.enable_debug_output:
                            self.get_logger().debug(f"Extracted and processed face crop from landmarks: {face_crop_balanced.shape}")
                        
                        return face_crop_balanced
                    else:
                        if self.enable_debug_output:
                            self.get_logger().debug("Landmark-based face crop is empty")
                        return face_crop
                else:
                    if self.enable_debug_output:
                        self.get_logger().debug(f"Invalid landmark-based dimensions: w={w}, h={h}")
            else:
                if self.enable_debug_output:
                    self.get_logger().debug(f"Not enough valid landmarks: {len(landmarks)} (need at least 4)")
            
            self.get_logger().warning(f"Failed to extract face crop for face {msg.face_id}: no valid bbox or landmarks")
            return None
            
        except Exception as e:
            self.get_logger().error(f"Failed to extract face crop: {e}")
            return None
    

    def _publish_batch_annotated_image(self, recognition_results: List):
        """Publish annotated image with recognition results for multiple faces as CompressedImage."""
        if not self.enable_image_output or not self.image_output_publisher or self.last_image is None:
            return
        try:
            # Create a copy of the image for annotation
            annotated_image = self.last_image.copy()
            if recognition_results:
                for landmarks_msg, (unique_id, confidence) in recognition_results:
                    self._draw_recognition_annotation(annotated_image, landmarks_msg, unique_id, confidence)
            # Encode as JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            small = cv2.resize(annotated_image, tuple(self.img_published_reshape_size), interpolation=cv2.INTER_AREA)
            success, encoded_image = cv2.imencode('.jpg', small, encode_params) # 3ms
            # success, encoded_image = cv2.imencode('.jpg', annotated_image) # 30-40ms
            if not success:
                self.get_logger().error("Failed to encode annotated image as JPEG")
                return
            compressed_msg = CompressedImage()
            compressed_msg.header = self.last_image_header if self.last_image_header else (recognition_results[0][0].header if recognition_results else Header())
            compressed_msg.format = 'jpeg'
            compressed_msg.data = encoded_image.tobytes()
            self.image_output_publisher.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish batch annotated image: {e}")
    
    def _publish_clean_image(self):
        """Publish the original image without any annotations when no faces are detected as CompressedImage."""
        if not self.enable_image_output or not self.image_output_publisher or self.last_image is None:
            return
        try:
            # Encode as JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            small = cv2.resize(self.last_image, tuple(self.img_published_reshape_size), interpolation=cv2.INTER_AREA)
            success, encoded_image = cv2.imencode('.jpg', small, encode_params) # 3ms
            # success, encoded_image = cv2.imencode('.jpg', annotated_image) # 30-40ms
            if not success:
                self.get_logger().error("Failed to encode clean image as JPEG")
                return
            compressed_msg = CompressedImage()
            compressed_msg.header = self.last_image_header if hasattr(self, 'last_image_header') and self.last_image_header else Header()
            if not compressed_msg.header.stamp.sec and not compressed_msg.header.stamp.nanosec:
                compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.format = 'jpeg'
            compressed_msg.data = encoded_image.tobytes()
            self.image_output_publisher.publish(compressed_msg)
            if self.enable_debug_output:
                self.get_logger().debug("Published clean compressed image (no faces detected)")
        except Exception as e:
            self.get_logger().error(f"Failed to publish clean compressed image: {e}")

    def _draw_recognition_annotation(self, image: np.ndarray, landmarks_msg, unique_id: Optional[str], confidence: float):
        """Draw recognition annotation on the image."""
        try:
            # Get face bounding box from bbox_xyxy (now NormalizedRegionOfInterest2D type)
            if hasattr(landmarks_msg.bbox_xyxy, 'xmin') and landmarks_msg.bbox_confidence > 0:
                # Convert normalized coordinates to pixel coordinates
                x1_norm, y1_norm = landmarks_msg.bbox_xyxy.xmin, landmarks_msg.bbox_xyxy.ymin
                x2_norm, y2_norm = landmarks_msg.bbox_xyxy.xmax, landmarks_msg.bbox_xyxy.ymax
                
                # Convert normalized coordinates to pixel coordinates
                x = int(x1_norm * landmarks_msg.width)
                y = int(y1_norm * landmarks_msg.height)
                w = int((x2_norm - x1_norm) * landmarks_msg.width)
                h = int((y2_norm - y1_norm) * landmarks_msg.height)
                
                # Draw face bounding box
                color = (0, 255, 0) if unique_id and unique_id != "unknown" else (0, 0, 255)  # Green for recognized, red for unknown
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Prepare text for unique ID
                if unique_id and unique_id != "unknown":
                    text = f"{unique_id} ({confidence:.2f})"
                else:
                    text = "Unknown"
                
                # Draw text background for unique ID
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Background rectangle for unique ID text
                cv2.rectangle(image, (x, y - text_height - 10), (x + text_width + 10, y), color, -1)
                
                # Draw unique ID text
                cv2.putText(image, text, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)
                
                # Prepare and draw face ID (original detection ID) text
                face_id_text = f"Face: {landmarks_msg.face_id}"
                (face_id_text_width, face_id_text_height), _ = cv2.getTextSize(face_id_text, font, font_scale, thickness)
                
                # Background rectangle for face ID text
                cv2.rectangle(image, (x, y + h + 10), (x + face_id_text_width + 10, y + h + 10 + face_id_text_height), color, -1)
                
                # Draw face ID text
                cv2.putText(image, face_id_text, (x + 5, y + h + 10 + face_id_text_height - 5), font, font_scale, (0, 0, 0), thickness)
            else:
                # Fallback: estimate position from landmarks
                landmarks = []
                for landmark in landmarks_msg.landmarks:
                    if landmark.c > 0:  # Valid landmark
                        # Convert normalized coordinates to pixel coordinates
                        x_pixel = int(landmark.x * landmarks_msg.width)
                        y_pixel = int(landmark.y * landmarks_msg.height)
                        landmarks.append([x_pixel, y_pixel])
                
                if len(landmarks) >= 2:
                    landmarks = np.array(landmarks)
                    x_min, y_min = np.min(landmarks, axis=0)
                    x_max, y_max = np.max(landmarks, axis=0)
                    
                    # Draw minimal annotation
                    color = (0, 255, 0) if unique_id and unique_id != "unknown" else (0, 0, 255)
                    text = f"{unique_id} ({confidence:.2f})" if unique_id and unique_id != "unknown" else "Unknown"
                    cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        except Exception as e:
            self.get_logger().error(f"Failed to draw recognition annotation: {e}")
    
    def destroy_node(self):
        """Clean up when node is destroyed."""
        self.get_logger().info("Destroy node")
        # Save identity database before shutdown
        if self.identity_manager and hasattr(self.identity_manager, 'save_identity_database'):
            try:
                self.identity_manager.save_identity_database()
                self.get_logger().debug("Identity database saved")
            except Exception as e:
                self.get_logger().error(f"Failed to save identity database: {e}")
        else:
            self.get_logger().info("No identity manager to save database from")
        super().destroy_node()


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RECOGNITION node.")
    except Exception as e:
        print(f"Error in face recognition node: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
