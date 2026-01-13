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
from collections import defaultdict
import time

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

from .enhanced_lip_movement_detector import EnhancedLipMovementDetector


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
        
        # Initialize enhanced lip movement detector
        self.lip_detector = EnhancedLipMovementDetector(
            window_size=self.window_size,
            movement_threshold=self.movement_threshold,
            speaking_threshold=self.speaking_threshold,
            temporal_smoothing=self.temporal_smoothing,
            min_frames_for_detection=self.min_frames_for_detection,
            use_full_landmarks=self.use_full_landmarks,
            rnn_enabled=self.rnn_enabled,
            logger=self.get_logger()
        )
        
        self.get_logger().info(f"Enhanced LipMovementDetector initialized: window_size={self.window_size}, movement_threshold={self.movement_threshold}, speaking_threshold={self.speaking_threshold}, use_full_landmarks={self.use_full_landmarks}, rnn_enabled={self.rnn_enabled}")
        
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
        
        # Setup QoS profiles
        self.qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        
        # Setup subscribers and publishers
        self._setup_topics()
        
        # Periodic cleanup timer
        cleanup_period = 5.0  # Clean up old identities every 5 seconds
        self.cleanup_timer = self.create_timer(cleanup_period, self._cleanup_old_identities)
        
        # Create timer for periodic debug info if debug is enabled
        if self.enable_debug_output:
            self.debug_timer = self.create_timer(10.0, self._debug_status_callback)
        
        self.get_logger().info("Visual Speech Activity Node initialized")
        self.get_logger().info(f"ROS4HRI mode: {'with_id' if self.ros4hri_with_id else 'array'}")
        self.get_logger().info(f"Face recognition mode: {'enabled' if self.use_face_recognition else 'disabled (face_id only)'}")
        self.get_logger().info(f"Window size: {self.window_size} frames")
        self.get_logger().info(f"Speaking threshold: {self.speaking_threshold}")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Input/Output topics
        self.declare_parameter('recognition_input_topic', '/humans/faces/recognized')
        self.declare_parameter('landmarks_input_topic', '/humans/faces/detected')
        self.declare_parameter('output_topic', '/humans/faces/speaking')
        
        # ROS4HRI mode parameter
        self.declare_parameter('ros4hri_with_id', False)  # Default to array mode
        
        # Lip movement detection parameters
        self.declare_parameter('window_size', 20)  # Number of frames for temporal analysis
        self.declare_parameter('movement_threshold', 0.02)  # Minimum MAR variation
        self.declare_parameter('speaking_threshold', 0.5)  # Confidence threshold for speaking
        self.declare_parameter('temporal_smoothing', True)  # Apply temporal smoothing
        self.declare_parameter('min_frames_for_detection', 5)  # Min frames before detection
        
        # Enhanced detector parameters
        self.declare_parameter('use_full_landmarks', True)  # Use full 68-point landmarks when available
        self.declare_parameter('rnn_enabled', True)  # Enable RNN-based classification
        
        # Face recognition dependency parameter
        self.declare_parameter('use_face_recognition', True)  # Use face recognition for robust tracking
        
        # Debug parameters
        self.declare_parameter('enable_debug_output', False)
    
    def _get_parameters(self):
        """Get parameters from ROS2 parameter server."""
        # Topics
        self.recognition_input_topic = self.get_parameter('recognition_input_topic').get_parameter_value().string_value
        self.landmarks_input_topic = self.get_parameter('landmarks_input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # ROS4HRI mode
        self.ros4hri_with_id = self.get_parameter('ros4hri_with_id').get_parameter_value().bool_value
        
        # Detection parameters
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.movement_threshold = self.get_parameter('movement_threshold').get_parameter_value().double_value
        self.speaking_threshold = self.get_parameter('speaking_threshold').get_parameter_value().double_value
        self.temporal_smoothing = self.get_parameter('temporal_smoothing').get_parameter_value().bool_value
        self.min_frames_for_detection = self.get_parameter('min_frames_for_detection').get_parameter_value().integer_value
        
        # Enhanced detector parameters
        self.use_full_landmarks = self.get_parameter('use_full_landmarks').get_parameter_value().bool_value
        self.rnn_enabled = self.get_parameter('rnn_enabled').get_parameter_value().bool_value
        
        # Face recognition dependency
        self.use_face_recognition = self.get_parameter('use_face_recognition').get_parameter_value().bool_value
        
        # Debug
        self.enable_debug_output = self.get_parameter('enable_debug_output').get_parameter_value().bool_value
    
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
            
            self.get_logger().info("Subscribed to /humans/faces/tracked for dynamic per-ID topics")
            
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
                self.get_logger().info(f"Subscribed to {self.recognition_input_topic} for face recognition")
            else:
                self.recognition_array_subscriber = None
                self.get_logger().info("Face recognition disabled - working with face_id only")
            
            self.speaking_array_publisher = self.create_publisher(
                FacialRecognitionArray,
                self.output_topic,
                self.qos_profile
            )
            
            self.get_logger().info(f"Subscribed to {self.landmarks_input_topic} (array mode)")
            self.get_logger().info(f"Subscribed to {self.recognition_input_topic} (array mode)")
            self.get_logger().info(f"Publishing to {self.output_topic} (array mode)")
            
            # No per-ID subscribers/publishers
            self.landmarks_subscribers = {}
            self.recognition_subscribers = {}
            self.speaking_publishers = {}
    
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
                self.get_logger().info(f"Added tracked face: {face_id}")
        
        # Remove old face IDs
        removed_ids = self.tracked_face_ids - current_ids
        for face_id in removed_ids:
            self._remove_per_id_subscribers(face_id)
            self.tracked_face_ids.discard(face_id)
            self.get_logger().info(f"Removed tracked face: {face_id}")
    
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
        
        self.get_logger().info(f"Created subscriber for topic: {landmarks_topic}")
        if self.enable_debug_output:
            self.get_logger().debug(f"Subscribed to {landmarks_topic} with QoS profile")
    
    def _remove_per_id_subscribers(self, face_id: str):
        """Remove per-ID subscribers for a face that's no longer tracked."""
        if face_id in self.landmarks_subscribers:
            self.destroy_subscription(self.landmarks_subscribers[face_id])
            del self.landmarks_subscribers[face_id]
        
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
    
    def _create_per_id_recognition_subscriber(self, recognized_face_id: str):
        """Create per-ID subscriber for recognition messages."""
        if recognized_face_id not in self.recognition_subscribers:
            recognition_topic = f'/humans/faces/{recognized_face_id}/recognition'
            self.recognition_subscribers[recognized_face_id] = self.create_subscription(
                FacialRecognition,
                recognition_topic,
                lambda msg, rid=recognized_face_id: self._recognition_per_id_callback(msg, rid),
                self.qos_profile
            )
            
            if self.enable_debug_output:
                self.get_logger().info(f"Subscribed to {recognition_topic}")
    
    def _recognition_per_id_callback(self, msg: FacialRecognition, recognized_face_id: str):
        """Callback for per-ID facial recognition."""
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
        for landmarks in msg.faces:
            self.latest_landmarks[landmarks.face_id] = landmarks
        
        if self.enable_debug_output:
            self.get_logger().info(f"Received {len(msg.faces)} facial landmarks")
        
        # If face recognition is disabled, process landmarks directly
        if not self.use_face_recognition:
            self._process_landmarks_without_recognition(msg.faces)
    
    def _recognition_array_callback(self, msg: FacialRecognitionArray):
        """
        Callback for facial recognition array.
        
        This triggers the speaking detection process for all recognized faces.
        """
        start_time = time.time()
        
        if len(msg.recognitions) == 0:
            # Publish empty result
            self._publish_speaking_array([])
            return
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Processing {len(msg.recognitions)} recognition(s)")
        
        # Update mappings and process each recognition
        speaking_recognitions = []
        
        for recognition in msg.recognitions:
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
            self.get_logger().info(f"Avg processing time: {avg_time*1000:.2f}ms")
    
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
        
        # Convert landmarks to list of (x, y) tuples
        landmarks_list = self._extract_landmark_coordinates(landmarks_msg)
        
        # Detect speaking
        is_speaking, speaking_confidence = self.lip_detector.detect_speaking(
            recognition.recognized_face_id,
            landmarks_list
        )
        
        # Create extended recognition message
        extended_recognition = self._copy_recognition_with_speaking(
            recognition, is_speaking, speaking_confidence
        )
        
        if self.enable_debug_output:
            self.get_logger().debug(
                f"Face {recognition.recognized_face_id}: "
                f"speaking={is_speaking}, confidence={speaking_confidence:.3f}"
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
        self._publish_speaking_per_id(speaking_recognition, recognized_face_id)
    
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
            
            # Use face_id as the identity for lip movement tracking
            # (less robust than recognized_face_id but works without face recognition)
            is_speaking, speaking_confidence = self.lip_detector.detect_speaking(
                face_id,  # Use face_id directly as identity
                landmarks_coords
            )
            
            # Create a FacialRecognition message using face_id
            recognition = FacialRecognition()
            recognition.header = landmarks_msg.header
            recognition.face_id = face_id
            recognition.recognized_face_id = face_id  # Same as face_id when no recognition
            recognition.confidence = 1.0  # Full confidence since no recognition uncertainty
            
            # Set speaking fields if they exist
            if hasattr(recognition, 'is_speaking'):
                recognition.is_speaking = is_speaking
                recognition.speaking_confidence = speaking_confidence
            
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
        
        if self.enable_debug_output:
            self.get_logger().debug(f"Extracted {len(landmarks_coords)} landmark coordinates for face {face_id}")
        
        # Use face_id as the identity for lip movement tracking
        is_speaking, speaking_confidence = self.lip_detector.detect_speaking(
            face_id,  # Use face_id directly as identity
            landmarks_coords
        )
        
        # Create a FacialRecognition message using face_id
        recognition = FacialRecognition()
        recognition.header = landmarks_msg.header
        recognition.face_id = face_id
        recognition.recognized_face_id = face_id  # Same as face_id when no recognition
        recognition.confidence = 1.0  # Full confidence since no recognition uncertainty
        
        # Set speaking fields if they exist
        if hasattr(recognition, 'is_speaking'):
            recognition.is_speaking = is_speaking
            recognition.speaking_confidence = speaking_confidence
        
        # Publish per-ID result
        self._publish_speaking_per_id(recognition, face_id)
        
        if self.enable_debug_output:
            self.get_logger().debug(
                f"Face {face_id} (no recognition, per-ID): "
                f"speaking={is_speaking}, confidence={speaking_confidence:.3f}"
            )
    
    def _debug_status_callback(self):
        """Periodic debug status callback."""
        if not self.enable_debug_output:
            return
        
        self.get_logger().debug(f"Active tracked faces: {list(self.tracked_face_ids)}")
        self.get_logger().debug(f"Active landmark subscribers: {list(self.landmarks_subscribers.keys())}")
        self.get_logger().debug(f"Latest landmarks cache: {list(self.latest_landmarks.keys())}")
        
        # Show detector status
        if hasattr(self, 'detectors'):
            self.get_logger().debug(f"Active detectors: {list(self.detectors.keys())}")
        elif hasattr(self, 'lip_detector'):
            self.get_logger().debug("Using single shared detector")
    
    def _extract_landmark_coordinates(self, landmarks_msg: FacialLandmarks) -> List[Tuple[float, float]]:
        """
        Extract landmark coordinates from FacialLandmarks message.
        
        Converts normalized coordinates to pixel coordinates for processing.
        
        Args:
            landmarks_msg: FacialLandmarks message
            
        Returns:
            List of (x, y) tuples in pixel coordinates
        """
        width = landmarks_msg.width
        height = landmarks_msg.height
        
        coordinates = []
        for landmark in landmarks_msg.landmarks:
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * width
            y = landmark.y * height
            coordinates.append((x, y))
        
        return coordinates
    
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
        extended_recognition = FacialRecognition()
        extended_recognition.header = recognition.header
        extended_recognition.face_id = recognition.face_id
        extended_recognition.recognized_face_id = recognition.recognized_face_id
        extended_recognition.confidence = recognition.confidence
        
        # Add speaking fields
        extended_recognition.is_speaking = is_speaking
        extended_recognition.speaking_confidence = speaking_confidence
        
        return extended_recognition
    
    # -------------------------------------------------------------------------
    #                    Publishing Methods
    # -------------------------------------------------------------------------
    
    def _publish_speaking_array(self, speaking_recognitions: List[FacialRecognition]):
        """Publish speaking detection results as array."""
        msg = FacialRecognitionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.recognitions = speaking_recognitions
        
        self.speaking_array_publisher.publish(msg)
        
        if self.enable_debug_output:
            speaking_count = sum(1 for r in speaking_recognitions if r.is_speaking)
            self.get_logger().debug(
                f"Published {len(speaking_recognitions)} recognitions, "
                f"{speaking_count} speaking"
            )
    
    def _publish_speaking_per_id(self, speaking_recognition: FacialRecognition, recognized_face_id: str):
        """Publish speaking detection result for a single face ID."""
        # Create publisher if it doesn't exist
        if recognized_face_id not in self.speaking_publishers:
            topic = f'/humans/faces/{recognized_face_id}/speaking'
            self.speaking_publishers[recognized_face_id] = self.create_publisher(
                FacialRecognition,
                topic,
                self.qos_profile
            )
            if self.enable_debug_output:
                self.get_logger().debug(f"Created publisher for {topic}")
        
        # Publish
        self.speaking_publishers[recognized_face_id].publish(speaking_recognition)
    
    # -------------------------------------------------------------------------
    #                    Cleanup Methods
    # -------------------------------------------------------------------------
    
    def _cleanup_old_identities(self):
        """Periodic cleanup of old identities from buffers."""
        # Get list of active recognized_face_ids
        active_ids = list(self.latest_recognition.keys())
        
        # Clean up lip detector buffers
        self.lip_detector.cleanup_old_identities(active_ids)
        
        # Clean up local caches (keep last 100 recognitions to avoid unbounded growth)
        if len(self.latest_recognition) > 100:
            # Sort by last access (we don't track this, so just keep most recent)
            # In a production system, you'd want to track last access time
            items = list(self.latest_recognition.items())
            self.latest_recognition = dict(items[-100:])


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
