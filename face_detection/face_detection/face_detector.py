#!/usr/bin/env python3
"""
Face Detection ROS2 Node

This node provides face detection capabilities using YOLO face detection.
It subscribes to RGB images and publishes ros4hri FacialLandmarks messages.

Input: sensor_msgs/Image (RGB)
Output: hri_msgs/FacialLandmarks (face bounding boxes and 5 key landmarks)

The node supports:
- YOLO Face Detection (5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth)
- Configurable detection parameters via launch file
- Optional image visualization output
"""
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from hri_msgs.msg import FacialLandmarks, FacialLandmarksArray, NormalizedPointOfInterest2D, NormalizedRegionOfInterest2D, IdsList
from std_msgs.msg import Header
from cv_bridge import CvBridge
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

from .yolo_face_detector import YoloFaceDetector


class FaceDetectorNode(Node):
    """
    ROS2 node for face detection using YOLO.
    
    This node processes RGB images and outputs face bounding boxes with 5 key landmarks
    in ros4hri FacialLandmarks format.
    """
    
    def __init__(self):
        super().__init__('face_detector')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Timing statistics
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.max_processing_time = 0.0
        self.min_processing_time = float('inf')
        
        # Declare parameters
        self._declare_parameters()
        
        # Get parameters
        self._get_parameters()
        
        # Initialize detector backend
        self.detector = None
        self._initialize_detector()

        # Initialize image storage variables (copied from perception node)
        self.latest_color_image_msg = None
        self.color_image_processed = False
        self.latest_color_image_timestamp = None
        
        # Setup QoS profiles (copied from perception node)
        sensor_qos = QoSProfile(
            depth=1,  # Keep only the latest image
            # reliability=QoSReliabilityPolicy.BEST_EFFORT,
            # durability=DurabilityPolicy.VOLATILE,
            # # history=QoSHistoryPolicy.KEEP_LAST,
        )
    
        # Create image subscribers - choose between compressed and regular image
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
            self.get_logger().info(f"Using regular image topic: {self.input_topic}")
            self.color_sub = self.create_subscription(
                Image, 
                self.input_topic, 
                self._store_latest_rgb, 
                sensor_qos
            )
        
        # Add a counter for received images
        self.image_count = 0
        
        # Create publisher based on mode - only one publisher per topic
        # (ros4hri_with_id is already set in _get_parameters())
        if self.ros4hri_with_id:
            # ROS4HRI with ID mode: Create per-ID publishers dynamically
            # Dictionary to store publishers for each face ID: {face_id: {'roi': Publisher, ...}}
            self.facial_landmarks_publishers = {}  # {face_id: {'roi': Publisher, ...}}
            self.tracked_face_ids = set()  # Set of currently tracked face IDs
            
            # Publisher for tracked faces list
            self.tracked_faces_publisher = self.create_publisher(
                IdsList,
                '/humans/faces/tracked',
                10
            )
            self.facial_landmarks_publisher = None
        else:
            # ROS4HRI array mode: Publish all faces in one FacialLandmarksArray message
            self.facial_landmarks_publisher = self.create_publisher(
                FacialLandmarksArray,
                self.output_topic,
                10)
            self.facial_landmarks_publishers = {}
            self.tracked_face_ids = set()
            self.tracked_faces_publisher = None
        
        # Create image publisher for visualization
        self.image_publisher = None
        if self.enable_image_output:
            self.image_publisher = self.create_publisher(
                Image,
                self.output_image_topic,
                 10)

        # Timer for periodic inference (copied from perception node pattern)
        timer_period = 1.0 / self.processing_rate_hz  # Use processing_rate_hz parameter
        self.inference_timer = self.create_timer(
            timer_period, 
            self.inference_timer_callback
        )
        
        self.get_logger().debug(f"Face Detector Node initialized")
        self.get_logger().debug(f"Input topic: {self.compressed_topic if self.compressed_topic and self.compressed_topic.strip() else self.input_topic}")
        self.get_logger().debug(f"Output topic: {self.output_topic}")
        if self.enable_image_output:
            self.get_logger().debug(f"Output image topic: {self.output_image_topic}")
        else:
            self.get_logger().debug("Image output disabled")

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
        # self.get_logger().debug("Compressed color image received.")

    # -------------------------------------------------------------------------
    #                         Timer Callback for Inference
    # -------------------------------------------------------------------------
    def inference_timer_callback(self):
        """
        Regular callback for continuous inference mode.

        Triggered by the timer at the configured frequency.
        Acquires the latest images, processes them, and publishes results.
        """
        
        start_time = self.get_clock().now()

        color_msg = self.latest_color_image_msg
        color_image_processed = self.color_image_processed

        if color_msg is None:
            self.get_logger().warning("No data received from color camera")
            return  # No data available yet
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

        # Increment counters
        self.image_count += 1
        self.frame_count += 1

        if self.detector is None:
            self.get_logger().warn("Face detector not initialized, skipping image")
            return
            
        if self.enable_debug_output:
            self.get_logger().debug(f"Processing image: {cv_image.shape}, dtype: {cv_image.dtype}")
        
        # Run face detection
        detection_results = self.detector.detect(cv_image)
        
        # Debug the detection results structure
        if self.enable_debug_output:
            self.get_logger().debug(f"Raw detection_results keys: {list(detection_results.keys())}")
            self.get_logger().debug(f"Raw detection_results: {detection_results}")
        
        # Log detection results conditionally
        num_faces = len(detection_results.get('faces', []))
        if self.enable_debug_output:
            if self.image_count % 30 == 1 or num_faces > 0:  # Log every 30 images OR when faces detected
                self.get_logger().debug(f"Detection results: {num_faces} face(s) detected")
                if num_faces > 0:
                    self.get_logger().debug(f"First face: {detection_results.get('faces', [])[0] if detection_results.get('faces') else 'None'}")
            
            # Log when faces are detected for debugging
            if num_faces > 0:
                self.get_logger().debug(f"[DEBUG] Face detection successful! Found {num_faces} faces")
                self.get_logger().debug(f"[DEBUG] Detection keys: {list(detection_results.keys())}")
                for key, value in detection_results.items():
                    self.get_logger().debug(f"[DEBUG] {key}: {len(value) if isinstance(value, list) else value}")
            elif self.image_count % 60 == 1:  # Log "no faces" less frequently
                self.get_logger().debug(f"[DEBUG] No faces detected in image #{self.image_count}")
        
        if self.enable_debug_output:
            if num_faces > 0:
                self.get_logger().debug(f"Detected {num_faces} face(s)")
                # Debug face coordinates
                for i, face in enumerate(detection_results.get('faces', [])):
                    self.get_logger().debug(f"Face {i}: {face}")
            else:
                # Log when no faces are detected (for debugging)
                self.get_logger().debug("No faces detected in this frame")
        
        # Convert to ROS messages and publish
        facial_landmarks_msgs = self._convert_to_facial_landmarks_msgs(
            detection_results, color_msg.header, cv_image.shape)
        
        # Log message conversion results (only if debug enabled)
        if self.enable_debug_output and num_faces > 0:
            self.get_logger().debug(f"[DEBUG] Converted {len(facial_landmarks_msgs)} faces to ROS messages")
        
        # Publish faces based on mode
        if facial_landmarks_msgs:
            if self.ros4hri_with_id:
                # ROS4HRI with ID mode: Publish individual fields to per-ID topics
                # Following ROS4HRI standard: /humans/faces/<faceID>/roi, etc.
                # All messages from the same frame share the same timestamp for synchronization
                frame_timestamp = color_msg.header.stamp
                current_frame_face_ids = set()
                
                for facial_landmarks_msg in facial_landmarks_msgs:
                    face_id = facial_landmarks_msg.face_id
                    current_frame_face_ids.add(face_id)
                    
                    # Create publishers for this face ID if they don't exist
                    if face_id not in self.facial_landmarks_publishers:
                        self.facial_landmarks_publishers[face_id] = {}
                        
                        # Publisher for full FacialLandmarks message
                        detected_topic = f'/humans/faces/{face_id}/detected'
                        self.facial_landmarks_publishers[face_id]['detected'] = self.create_publisher(
                            FacialLandmarks,
                            detected_topic,
                            10
                        )
                        self.get_logger().info(f"Created FacialLandmarks publisher for face ID: {detected_topic}")
                        
                        # Publisher for ROI (bounding box) - individual field
                        roi_topic = f'/humans/faces/{face_id}/roi'
                        self.facial_landmarks_publishers[face_id]['roi'] = self.create_publisher(
                            NormalizedRegionOfInterest2D,
                            roi_topic,
                            10
                        )
                        self.get_logger().info(f"Created ROI publisher for face ID: {roi_topic}")
                    
                    # Ensure all messages from the same frame have the same timestamp
                    facial_landmarks_msg.header.stamp = frame_timestamp
                    
                    # Publish full FacialLandmarks message to /humans/faces/<faceID>/detected
                    self.facial_landmarks_publishers[face_id]['detected'].publish(facial_landmarks_msg)
                    
                    # Publish ROI (bounding box) to /humans/faces/<faceID>/roi (individual field)
                    roi_msg = NormalizedRegionOfInterest2D()
                    roi_msg.header = facial_landmarks_msg.header
                    roi_msg.header.stamp = frame_timestamp  # Same timestamp for all messages from same frame
                    roi_msg.xmin = facial_landmarks_msg.bbox_xyxy.xmin
                    roi_msg.ymin = facial_landmarks_msg.bbox_xyxy.ymin
                    roi_msg.xmax = facial_landmarks_msg.bbox_xyxy.xmax
                    roi_msg.ymax = facial_landmarks_msg.bbox_xyxy.ymax
                    roi_msg.c = facial_landmarks_msg.bbox_confidence
                    
                    self.facial_landmarks_publishers[face_id]['roi'].publish(roi_msg)
                    
                    if self.enable_debug_output:
                        self.get_logger().debug(f"[ROS PUBLISH] Published FacialLandmarks for face_id={face_id} to /humans/faces/{face_id}/detected with timestamp={frame_timestamp}")
                        self.get_logger().debug(f"[ROS PUBLISH] Published ROI for face_id={face_id} to /humans/faces/{face_id}/roi with timestamp={frame_timestamp}")
                
                # Update tracked faces list
                # Remove face IDs that are no longer present
                removed_ids = self.tracked_face_ids - current_frame_face_ids
                for face_id in removed_ids:
                    if face_id in self.facial_landmarks_publishers:
                        # Note: ROS2 doesn't allow destroying publishers, but we can stop using them
                        # The topic will remain but won't receive new messages
                        if self.enable_debug_output:
                            self.get_logger().debug(f"Face ID {face_id} no longer tracked")
                
                # Update tracked face IDs set
                self.tracked_face_ids = current_frame_face_ids
                
                # Publish tracked faces list
                tracked_list_msg = IdsList()
                tracked_list_msg.header = color_msg.header
                tracked_list_msg.ids = list(self.tracked_face_ids)
                self.tracked_faces_publisher.publish(tracked_list_msg)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"[ROS PUBLISH] Published tracked faces list: {list(self.tracked_face_ids)}")
            else:
                # ROS4HRI array mode: Publish all faces in one FacialLandmarksArray message
                facial_landmarks_array = FacialLandmarksArray()
                facial_landmarks_array.header = color_msg.header
                facial_landmarks_array.ids = facial_landmarks_msgs
                
                self.facial_landmarks_publisher.publish(facial_landmarks_array)
                if self.enable_debug_output:
                    self.get_logger().debug(f"[ROS PUBLISH] Published FacialLandmarksArray with {len(facial_landmarks_msgs)} faces")
        
        # Error logging (always log errors)
        if len(facial_landmarks_msgs) == 0 and num_faces > 0:
            self.get_logger().error(f"[ERROR] Detected {num_faces} faces but converted 0 messages!")
        elif len(facial_landmarks_msgs) == 0 and self.enable_debug_output:
            if self.image_count % 60 == 1:
                self.get_logger().debug("No faces detected, no messages published")
        
        # Publish visualization if enabled
        if self.enable_image_output and self.image_publisher is not None:
            self._publish_image_with_faces(cv_image, detection_results, color_msg.header)

        # Calculate and log timing information
        end_time = self.get_clock().now()
        processing_time = (end_time - start_time).nanoseconds / 1e6  # Convert to milliseconds
        
        # Update timing statistics
        self.total_processing_time += processing_time
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.min_processing_time = min(self.min_processing_time, processing_time)
        
        # Log timing every 100 frames or when debug is enabled
        if self.frame_count % 100 == 0 or self.enable_debug_output:
            avg_time = self.total_processing_time / self.frame_count
            self.get_logger().info(
                f"[TIMING] Face Detection - Frame #{self.frame_count}: "
                f"Current: {processing_time:.2f}ms, "
                f"Avg: {avg_time:.2f}ms, "
                f"Min: {self.min_processing_time:.2f}ms, "
                f"Max: {self.max_processing_time:.2f}ms"
            )
        
    def _declare_parameters(self):
        """Declare ROS2 parameters with default values."""
        self.declare_parameter('compressed_topic', '')
        self.declare_parameter('input_topic', '/camera/color/image_rect_raw')
        self.declare_parameter('output_topic', '/humans/faces/detected')
        self.declare_parameter('output_image_topic', '/humans/faces/detected/annotated_img')
        
        # Processing rate parameter (copied from perception node)
        self.declare_parameter('processing_rate_hz', 30.0)  # Default 10 Hz
        
        # YOLO Face Detection Parameters
        self.declare_parameter('model_path', 'weights/yolov8n-face.onnx')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.4)
        self.declare_parameter('device', 'cpu')
        
        # General parameters  
        self.declare_parameter('enable_debug_output', False)  # Disable debug by default
        self.declare_parameter('face_id_prefix', 'face_')
        
        # ROS4HRI mode parameter - when enabled, publishes per-ID messages instead of arrays
        self.declare_parameter('ros4hri_with_id', False)  # Default to array mode (ROS4HRI array)
        
        # BOXMOT tracking parameters
        self.declare_parameter('use_boxmot', False)
        self.declare_parameter('boxmot_tracker_type', 'bytetrack')
        self.declare_parameter('boxmot_reid_model', '')
        
        # Image visualization parameters
        self.declare_parameter('enable_image_output', True)
        self.declare_parameter('face_bbox_thickness', 2)
        self.declare_parameter('face_landmark_radius', 3)
        self.declare_parameter('face_bbox_color', [0, 255, 0])  # Green
        self.declare_parameter('face_landmark_color', [255, 0, 0])  # Blue
        
    def _get_parameters(self):
        """Get parameter values from ROS2 parameter server."""
        self.compressed_topic = self.get_parameter('compressed_topic').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        
        # Get debug parameter first since it's used in path resolution
        self.enable_debug_output = self.get_parameter('enable_debug_output').get_parameter_value().bool_value
        
        # YOLO parameters
        model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        
        # If model path is relative, make it relative to package source directory
        if not os.path.isabs(model_path_param):
            # Try to find the package source directory dynamically
            try:
                # Get the current file's directory and navigate to package root
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                # Navigate up from face_detection/face_detection/ to face_detection/
                package_src_dir = os.path.dirname(current_file_dir)
                package_src_dir = package_src_dir.replace('build', 'src') #save it in docker volume of ros2 package
                self.model_path = os.path.join(package_src_dir,"weights",model_path_param)

                if self.enable_debug_output:
                    self.get_logger().debug(f"Using package source directory: {package_src_dir}")
                    self.get_logger().debug(f"Model path resolved to: {self.model_path}")
                    
            except Exception as e:
                # Fallback to share directory if source directory detection fails
                self.get_logger().warn(f"Could not determine package source directory: {e}")
                self.get_logger().warn("Falling back to package share directory")
                package_share_directory = get_package_share_directory('face_detection')
                self.model_path = os.path.join(package_share_directory, model_path_param)
        else:
            self.model_path = model_path_param
            
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        
        self.face_id_prefix = self.get_parameter('face_id_prefix').get_parameter_value().string_value
        
        # BOXMOT parameters
        self.use_boxmot = self.get_parameter('use_boxmot').get_parameter_value().bool_value
        self.boxmot_tracker_type = self.get_parameter('boxmot_tracker_type').get_parameter_value().string_value
        self.boxmot_reid_model = self.get_parameter('boxmot_reid_model').get_parameter_value().string_value
        
        # Image visualization parameters
        self.enable_image_output = self.get_parameter('enable_image_output').get_parameter_value().bool_value
        self.face_bbox_thickness = self.get_parameter('face_bbox_thickness').get_parameter_value().integer_value
        self.face_landmark_radius = self.get_parameter('face_landmark_radius').get_parameter_value().integer_value
        self.face_bbox_color = self.get_parameter('face_bbox_color').get_parameter_value().integer_array_value
        self.face_landmark_color = self.get_parameter('face_landmark_color').get_parameter_value().integer_array_value
        
        # Get processing rate parameter
        self.processing_rate_hz = self.get_parameter('processing_rate_hz').get_parameter_value().double_value
        
        # Get ROS4HRI mode parameter
        self.ros4hri_with_id = self.get_parameter('ros4hri_with_id').get_parameter_value().bool_value
        
        # Log mode (ros4hri_with_id already set before creating publishers)
        if self.ros4hri_with_id:
            self.get_logger().info("ROS4HRI with ID mode enabled: Publishing individual FacialLandmarks messages per face ID")
        else:
            self.get_logger().info("ROS4HRI array mode enabled: Publishing FacialLandmarksArray messages")
        
    def _initialize_detector(self):
        """Initialize the face detection backend."""
        try:
            self.detector = YoloFaceDetector(
                logger=self.get_logger(),
                model_path=self.model_path,
                conf_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                device=self.device,
                debug=self.enable_debug_output,
                use_boxmot=self.use_boxmot,
                boxmot_tracker_type=self.boxmot_tracker_type,
                boxmot_reid_model=self.boxmot_reid_model
            )
            
            if self.detector.initialize():
                self.get_logger().debug("YOLO face detector initialized successfully")
            else:
                self.get_logger().error("Failed to initialize YOLO face detector")
                self.detector = None
                
        except Exception as e:
            self.get_logger().error(f"Failed to initialize face detector: {e}")
            self.detector = None
    
    def _convert_to_facial_landmarks_msgs(self, detection_results: Dict[str, Any], 
                                        original_header: Header, 
                                        image_shape: Tuple[int, int, int]) -> List[FacialLandmarks]:
        """
        Convert detection results to ros4hri FacialLandmarks messages.
        
        Args:
            detection_results: Detection results from YOLO face detector
            original_header: Original image header
            image_shape: Image shape (height, width, channels)
            
        Returns:
            List of FacialLandmarks messages
        """
        facial_landmarks_msgs = []
        height, width = image_shape[:2]
        
        faces = detection_results.get('faces', [])
        confidences = detection_results.get('confidences', [])
        landmarks = detection_results.get('landmarks', [])
        track_ids = detection_results.get('track_ids', list(range(len(faces))))  # Use track IDs if available, fallback to indices
        
        for i, (face_bbox, confidence, face_landmarks, track_id) in enumerate(zip(faces, confidences, landmarks, track_ids)):
            try:
                # Create FacialLandmarks message
                facial_landmarks_msg = FacialLandmarks()
                facial_landmarks_msg.header = original_header
                facial_landmarks_msg.face_id = f"{self.face_id_prefix}{track_id}"
                
                # Set image dimensions
                facial_landmarks_msg.height = height
                facial_landmarks_msg.width = width
                
                # Set bounding box information using NormalizedRegionOfInterest2D
                facial_landmarks_msg.bbox_confidence = float(confidence)
                x, y, w, h = face_bbox
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"Processing face {i}: bbox=({x}, {y}, {w}, {h}), confidence={confidence}")
                
                # Ensure coordinates are non-negative and within image bounds
                x = max(0, min(int(x), width - 1))
                y = max(0, min(int(y), height - 1))
                w = max(1, min(int(w), width - x))
                h = max(1, min(int(h), height - y))
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"Clamped face {i}: bbox=({x}, {y}, {w}, {h})")
                
                # Create NormalizedRegionOfInterest2D for bbox_xyxy
                bbox_roi = NormalizedRegionOfInterest2D()
                bbox_roi.header = original_header
                
                # Normalize coordinates to [0,1] range
                bbox_roi.xmin = float(x) / width
                bbox_roi.ymin = float(y) / height
                bbox_roi.xmax = float(x + w) / width
                bbox_roi.ymax = float(y + h) / height
                bbox_roi.c = float(confidence)
                
                facial_landmarks_msg.bbox_xyxy = bbox_roi
                facial_landmarks_msg.bbox_centroid = [float(x + w/2.0), float(y + h/2.0)]
                
                # Convert YOLO 5-point landmarks to ros4hri format
                facial_landmarks_msg.landmarks = self._convert_yolo_landmarks_to_ros4hri(
                    face_landmarks, width, height)
                
                if self.enable_debug_output:
                    self.get_logger().debug(f"Created FacialLandmarks message for face {i} with {len(facial_landmarks_msg.landmarks)} landmarks")
                
                facial_landmarks_msgs.append(facial_landmarks_msg)
                
            except Exception as e:
                self.get_logger().error(f"Error processing face {i}: {e}")
                continue
            
        return facial_landmarks_msgs
    
    def _convert_yolo_landmarks_to_ros4hri(self, yolo_landmarks: List[float], 
                                         image_width: int, image_height: int) -> List[NormalizedPointOfInterest2D]:
        """
        Convert YOLO 5-point landmarks to ros4hri FacialLandmarks format.
        
        YOLO provides 5 landmarks: [left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
                                   nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y]
        
        ros4hri FacialLandmarks expects 70 landmarks following dlib/OpenPose convention.
        We'll map the available 5 points to the corresponding ros4hri indices and set others to invalid.
        
        Args:
            yolo_landmarks: List of 10 values (5 points * 2 coordinates each)
            image_width: Image width for normalization
            image_height: Image height for normalization
            
        Returns:
            List of NormalizedPointOfInterest2D messages (70 landmarks)
        """
        # Initialize all 70 landmarks as invalid (confidence = 0.0, position = 0.0)
        landmarks = []
        for i in range(70):
            landmark = NormalizedPointOfInterest2D()
            landmark.x = 0.0
            landmark.y = 0.0
            landmark.c = 0.0  # Invalid confidence
            landmarks.append(landmark)
        
        # If we don't have the expected 10 values, return all invalid landmarks
        if len(yolo_landmarks) != 10:
            if self.enable_debug_output:
                self.get_logger().warn(f"Expected 10 landmark values, got {len(yolo_landmarks)}")
            return landmarks
        
        # Extract YOLO landmarks (pixel coordinates)
        left_eye_x, left_eye_y = yolo_landmarks[0], yolo_landmarks[1]
        right_eye_x, right_eye_y = yolo_landmarks[2], yolo_landmarks[3]
        nose_x, nose_y = yolo_landmarks[4], yolo_landmarks[5]
        left_mouth_x, left_mouth_y = yolo_landmarks[6], yolo_landmarks[7]
        right_mouth_x, right_mouth_y = yolo_landmarks[8], yolo_landmarks[9]
        
        # Normalize to [0, 1] range with bounds checking
        def normalize_point(x, y):
            # Clamp coordinates to image bounds
            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))
            return (x / image_width, y / image_height, 1.0)  # confidence = 1.0 for valid points
        
        # Map YOLO landmarks to ros4hri indices
        # Based on FacialLandmarks.msg constants:
        
        # Left eye (from face perspective, so it's the right eye in the image)
        # Use LEFT_EYE_INSIDE (42) as the main left eye point
        norm_x, norm_y, conf = normalize_point(left_eye_x, left_eye_y)
        landmarks[42].x = norm_x  # LEFT_EYE_INSIDE
        landmarks[42].y = norm_y
        landmarks[42].c = conf
        
        # Right eye (from face perspective, so it's the left eye in the image)  
        # Use RIGHT_EYE_INSIDE (39) as the main right eye point
        norm_x, norm_y, conf = normalize_point(right_eye_x, right_eye_y)
        landmarks[39].x = norm_x  # RIGHT_EYE_INSIDE
        landmarks[39].y = norm_y
        landmarks[39].c = conf
        
        # Nose tip
        # Use NOSE (30) as the main nose point
        norm_x, norm_y, conf = normalize_point(nose_x, nose_y)
        landmarks[30].x = norm_x  # NOSE
        landmarks[30].y = norm_y
        landmarks[30].c = conf
        
        # Left mouth corner  
        # Use MOUTH_OUTER_LEFT (54) as the left mouth point
        norm_x, norm_y, conf = normalize_point(left_mouth_x, left_mouth_y)
        landmarks[54].x = norm_x  # MOUTH_OUTER_LEFT
        landmarks[54].y = norm_y
        landmarks[54].c = conf
        
        # Right mouth corner
        # Use MOUTH_OUTER_RIGHT (48) as the right mouth point
        norm_x, norm_y, conf = normalize_point(right_mouth_x, right_mouth_y)
        landmarks[48].x = norm_x  # MOUTH_OUTER_RIGHT
        landmarks[48].y = norm_y
        landmarks[48].c = conf
        
        if self.enable_debug_output:
            valid_landmarks = sum(1 for lm in landmarks if lm.c > 0.0)
            self.get_logger().debug(f"Mapped {valid_landmarks}/70 landmarks from YOLO 5-point detection")
        
        return landmarks

    def _publish_image_with_faces(self, cv_image: np.ndarray, detection_results: Dict[str, Any], header: Header):
        """
        Publish image with face overlays for visualization.
        
        Args:
            cv_image: Original OpenCV image
            detection_results: Face detection results
            header: Original image header
        """
        try:
            # Create a copy for annotation
            annotated_image = cv_image.copy()
            
            faces = detection_results.get('faces', [])
            confidences = detection_results.get('confidences', [])
            landmarks = detection_results.get('landmarks', [])
            track_ids = detection_results.get('track_ids', list(range(len(faces))))  # Fallback to enumeration
            
            # Draw faces
            for i, (face_bbox, confidence, face_landmarks, track_id) in enumerate(zip(faces, confidences, landmarks, track_ids)):
                self._draw_face_on_image(annotated_image, face_bbox, face_landmarks, confidence, track_id)
            
            # Convert back to ROS Image and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = header
            self.image_publisher.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing annotated image: {e}")
    
    def _draw_face_on_image(self, image: np.ndarray, face_bbox: List[int], 
                           face_landmarks: List[float], confidence: float, face_id: int):
        """
        Draw a single face on the image with adaptive sizing.
        
        Args:
            image: OpenCV image to draw on
            face_bbox: Face bounding box [x, y, w, h]
            face_landmarks: Face landmarks (10 values: 5 points * 2 coordinates)
            confidence: Face detection confidence
            face_id: ID of the face for labeling
        """
        # Calculate adaptive sizes based on image dimensions
        img_height, img_width = image.shape[:2]
        base_size = min(img_width, img_height)
        
        # Adaptive thickness and sizes (scale with image size)
        adaptive_bbox_thickness = max(2, int(base_size * 0.003))  # 0.3% of image size
        adaptive_landmark_radius = max(3, int(base_size * 0.008))  # 0.8% of image size
        adaptive_line_thickness = max(2, int(base_size * 0.002))   # 0.2% of image size
        adaptive_font_scale = max(0.5, base_size * 0.001)          # Adaptive font size
        adaptive_font_thickness = max(1, int(base_size * 0.0015))  # Adaptive font thickness
        
        # Draw bounding box with adaptive thickness
        x, y, w, h = face_bbox
        bbox_color = tuple(int(c) for c in self.face_bbox_color)
        cv2.rectangle(image, (x, y), (x + w, y + h), bbox_color, adaptive_bbox_thickness)
        
        # Draw confidence label with adaptive font
        label = f"Face {face_id}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, adaptive_font_thickness)[0]
        label_padding = max(5, int(base_size * 0.01))
        
        # Background rectangle for label
        cv2.rectangle(image, 
                     (x, y - label_size[1] - label_padding * 2), 
                     (x + label_size[0] + label_padding, y), 
                     bbox_color, -1)
        
        # Text with adaptive size
        cv2.putText(image, label, 
                   (x + label_padding//2, y - label_padding), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   adaptive_font_scale, 
                   (0, 0, 0), 
                   adaptive_font_thickness)
        
        # Draw landmarks if available
        if len(face_landmarks) == 10:
            landmark_color = tuple(int(c) for c in self.face_landmark_color)
            
            # Draw the 5 key points with adaptive radius
            landmark_names = ["Left Eye", "Right Eye", "Nose", "Left Mouth", "Right Mouth"]
            for i in range(0, 10, 2):
                lm_x, lm_y = int(face_landmarks[i]), int(face_landmarks[i + 1])
                
                # Draw larger landmark circles
                cv2.circle(image, (lm_x, lm_y), adaptive_landmark_radius, landmark_color, -1)
                
                # Optional: Draw landmark labels for debugging
                if self.enable_debug_output and i//2 < len(landmark_names):
                    landmark_label = landmark_names[i//2]
                    cv2.putText(image, landmark_label, 
                               (lm_x + adaptive_landmark_radius + 2, lm_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               adaptive_font_scale * 0.7, 
                               landmark_color, 
                               max(1, adaptive_font_thickness - 1))
            
            # Draw lines connecting landmarks with adaptive thickness
            if len(face_landmarks) >= 10:
                # Connect eyes
                left_eye = (int(face_landmarks[0]), int(face_landmarks[1]))
                right_eye = (int(face_landmarks[2]), int(face_landmarks[3]))
                cv2.line(image, left_eye, right_eye, landmark_color, adaptive_line_thickness)
                
                # Connect mouth corners
                left_mouth = (int(face_landmarks[6]), int(face_landmarks[7]))
                right_mouth = (int(face_landmarks[8]), int(face_landmarks[9]))
                cv2.line(image, left_mouth, right_mouth, landmark_color, adaptive_line_thickness)
                
                # Connect nose to eyes (optional, creates face structure)
                nose = (int(face_landmarks[4]), int(face_landmarks[5]))
                cv2.line(image, nose, left_eye, landmark_color, max(1, adaptive_line_thickness - 1), cv2.LINE_AA)
                cv2.line(image, nose, right_eye, landmark_color, max(1, adaptive_line_thickness - 1), cv2.LINE_AA)
        
def main(args=None):
    """Main function to run the face detector node."""
    rclpy.init(args=args)
    
    try:
        face_detector_node = FaceDetectorNode()
        rclpy.spin(face_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
