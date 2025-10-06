#!/usr/bin/env python3
"""
Face Recognition Node for HRI Applications

This node subscribes to FacialLandmarks messages from face detection,
extracts face embeddings, performs identity clustering and temporal tracking,
and publishes FacialRecognition messages following the ros4hri standard.

The approach is 100% based on the EUT YOLO identity management system,
providing persistent identity tracking across changing track IDs.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import numpy as np
import cv2
import time
import os
from typing import Dict, List, Optional, Tuple, Any

try:
    from hri_msgs.msg import FacialLandmarks, FacialLandmarksArray, FacialRecognition
except ImportError:
    print("Warning: hri_msgs not found. Please install hri_msgs package.")
    FacialLandmarks = None
    FacialLandmarksArray = None
    FacialRecognition = None

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .face_embedding_extractor import create_face_embedding_extractor
from .identity_manager import IdentityManager


class FaceRecognitionNode(Node):
    """
    ROS2 node for face recognition using embedding-based identity management.
    
    Subscribes to FacialLandmarks messages and publishes FacialRecognition messages
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
        
        # Store latest image for processing
        self.last_image: Optional[np.ndarray] = None
        self.last_image_header: Optional[Header] = None
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.processed_messages = 0
        
        # Debug settings
        self.enable_debug_prints = False  # Will be set during initialization
        
        # Initialize QoS profiles
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Setup subscribers and publishers
        self._setup_topics()
        
        # Initialize face embedding extractor and identity manager
        self._initialize_components()
        
        self.get_logger().info("Face Recognition Node initialized")
    
    def _declare_parameters(self):
        """Declare ROS2 parameters."""
        # Input/Output topics
        self.declare_parameter('input_topic', '/people/faces/detected')
        self.declare_parameter('output_topic', '/people/faces/recognized')
        self.declare_parameter('image_input_topic', '/camera/color/image_rect_raw')
        
        # Image output parameters
        self.declare_parameter('enable_image_output', True)
        self.declare_parameter('image_output_topic', '/people/faces/recognized/image_with_recognition')
        
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
        self.declare_parameter('enable_debug_prints', True)  # Temporarily enable for debugging
        self.declare_parameter('identity_database_path', '')
        self.declare_parameter('use_ewma_for_mean', False)
        self.declare_parameter('ewma_alpha', 0.6)
        
        # Processing parameters
        self.declare_parameter('gaze_identity_exclusion_threshold', 0.5)
        
        # Receiver ID for hri_msgs
        self.declare_parameter('receiver_id', 'face_recognition')
    
    def _setup_topics(self):
        """Setup ROS2 subscribers and publishers."""
        # Get topic names from parameters
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        image_input_topic = self.get_parameter('image_input_topic').get_parameter_value().string_value
        
        # Subscriber for facial landmarks array
        self.landmarks_subscriber = self.create_subscription(
            FacialLandmarksArray,
            input_topic,
            self.landmarks_array_callback,
            self.qos_profile
        )
        
        # Subscriber for RGB images  
        self.image_subscriber = self.create_subscription(
            Image,
            image_input_topic, 
            self.image_callback,
            self.qos_profile
        )
        
        # Publisher for facial recognition results
        self.recognition_publisher = self.create_publisher(
            FacialRecognition,
            output_topic,
            self.qos_profile
        )
        
        # Image output publisher (optional)
        self.enable_image_output = self.get_parameter('enable_image_output').get_parameter_value().bool_value
        if self.enable_image_output:
            image_output_topic = self.get_parameter('image_output_topic').get_parameter_value().string_value
            self.image_output_publisher = self.create_publisher(
                Image,
                image_output_topic,
                self.qos_profile
            )
            self.get_logger().info(f"Image output enabled: {image_output_topic}")
        else:
            self.image_output_publisher = None
        
        self.get_logger().info(f"Subscribed to: {input_topic}")
        self.get_logger().info(f"Subscribed to: {image_input_topic}")
        self.get_logger().info(f"Publishing to: {output_topic}")
    
    def _initialize_components(self):
        """Initialize face embedding extractor and identity manager."""
        # Get parameters
        face_embedding_model = self.get_parameter('face_embedding_model').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        face_embedding_weights_name = self.get_parameter('face_embedding_weights_name').get_parameter_value().string_value
        
        # Build full weights path similar to YOLO approach
        weights_dir_path = None
        face_embedding_weights_path = None
        
        if weights_path:
            # Try to find the source directory first (for development)
            possible_paths = [
                # Source directory (development workspace)
                '/workspace/src/face_recognition/weights',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), weights_path),
                # Install directory (after colcon build)
                None  # Will be set below
            ]
            
            try:
                import ament_index_python.packages as ament_packages
                package_share = ament_packages.get_package_share_directory('face_recognition')
                possible_paths.append(os.path.join(package_share, weights_path))
            except:
                pass
            
            # Check each possible path
            for path in possible_paths:
                if path and os.path.exists(path):
                    weights_dir_path = path
                    self.get_logger().info(f"Weights directory found: {weights_dir_path}")
                    break
            
            # Fallback to relative path if none found
            if not weights_dir_path:
                weights_dir_path = weights_path
                self.get_logger().warn(f"Using relative weights path: {weights_dir_path}")
            
            # If specific weights filename is provided, build full path
            if face_embedding_weights_name:
                face_embedding_weights_path = os.path.join(weights_dir_path, face_embedding_weights_name)
                self.get_logger().info(f"Face embedding weights path: {face_embedding_weights_path}")
        
        # Initialize face embedding extractor
        try:
            self.face_embedding_extractor = create_face_embedding_extractor(
                model_name=face_embedding_model,
                device=device,
                weights_path=weights_dir_path,
                face_embedding_weights_path=face_embedding_weights_path
            )
            
            if self.face_embedding_extractor.is_available():
                self.get_logger().info(f"Face embedding extractor initialized: {face_embedding_model} on {device}")
                model_info = self.face_embedding_extractor.get_model_info()
                self.get_logger().info(f"Model info: {model_info}")
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
            debug_prints = self.get_parameter('enable_debug_prints').get_parameter_value().bool_value
            database_path = self.get_parameter('identity_database_path').get_parameter_value().string_value
            use_ewma = self.get_parameter('use_ewma_for_mean').get_parameter_value().bool_value
            ewma_alpha = self.get_parameter('ewma_alpha').get_parameter_value().double_value
            
            # Set debug prints flag
            self.enable_debug_prints = debug_prints
            
            self.identity_manager = IdentityManager(
                max_embeddings_per_identity=max_embeddings,
                similarity_threshold=similarity_thresh,
                track_identity_stickiness_margin=stickiness_margin,
                clustering_threshold=clustering_thresh,
                embedding_inclusion_threshold=embedding_inclusion_thresh,
                identity_timeout=identity_timeout,
                min_detections_for_stable_identity=min_detections,
                enable_debug_prints=debug_prints,
                identity_database_path=database_path if database_path else None,
                use_ewma_for_mean=use_ewma,
                ewma_alpha=ewma_alpha
            )
            
            self.get_logger().info("Identity manager initialized")
            self.get_logger().info(f"Parameters: similarity_threshold={similarity_thresh}, clustering_threshold={clustering_thresh}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize identity manager: {e}")
            return
    
    def image_callback(self, msg: Image):
        """Callback for RGB image messages."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_image = cv_image
            self.last_image_header = msg.header
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    

    
    def landmarks_array_callback(self, msg):
        """
        Callback for processing array of facial landmarks and computing face recognition for all faces.
        
        Args:
            msg: FacialLandmarksArray message containing multiple face landmarks
        """
        if not self.face_embedding_extractor or not self.identity_manager:
            self.get_logger().warning("Face recognition components not initialized")
            return
        
        if self.last_image is None:
            self.get_logger().warning("No image available for face recognition")
            return
            
        if not msg.ids:
            if self.enable_debug_prints:
                self.get_logger().debug('Received empty FacialLandmarksArray')
            return
        
        if self.enable_debug_prints:
            self.get_logger().debug(f'Processing FacialLandmarksArray with {len(msg.ids)} faces')
        
        start_time = time.time()
        
        try:
            # Process all faces in batch mode
            self._process_landmarks_array_batch(msg)
        
        except Exception as e:
            self.get_logger().error(f"Error in landmarks array callback: {e}")
        
        # Log processing time
        processing_time = (time.time() - start_time) * 1000
        self.total_processing_time += processing_time
        self.processed_messages += 1
        
        if self.processed_messages % 50 == 0:
            avg_time = self.total_processing_time / self.processed_messages
            self.get_logger().info(f"Average processing time: {avg_time:.2f}ms")
    

    
    def _process_landmarks_array_batch(self, msg):
        """Process array of facial landmarks in batch mode for better performance."""
        if not msg.ids:
            return
        
        # Extract face crops and face IDs for all faces
        if self.enable_debug_prints:
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
        
        if self.enable_debug_prints:
            crop_time = (time.time() - crop_start_time) * 1000
            self.get_logger().info(f"Face crop extraction took: {crop_time:.2f}ms for {len(msg.ids)} input faces, got {len(face_crops)} valid crops")
        
        if not face_crops:
            self.get_logger().warning("No valid face crops extracted from landmarks array")
            return
        
        # Check if face embedding extractor is available
        if not self.face_embedding_extractor.is_available():
            self.get_logger().error("Face embedding extractor is not available")
            return
        
        # Extract embeddings in batch
        if self.enable_debug_prints:
            embedding_start_time = time.time()
        
        embeddings = self.face_embedding_extractor.extract_embeddings_batch(face_crops)
        
        if self.enable_debug_prints:
            embedding_time = (time.time() - embedding_start_time) * 1000
            self.get_logger().info(f"Embedding extraction took: {embedding_time:.2f}ms for {len(face_crops)} faces")
        
        # Create face_embeddings dictionary for identity manager using face_id as key
        if self.enable_debug_prints:
            prep_start_time = time.time()
        
        face_embeddings = {}
        valid_indices = []
        
        for i, (face_id, embedding) in enumerate(zip(face_ids, embeddings)):
            if embedding is not None:
                face_embeddings[face_id] = embedding
                valid_indices.append(i)
            else:
                self.get_logger().warning(f"Failed to extract embedding for face {face_ids[i]}")
        
        if self.enable_debug_prints:
            prep_time = (time.time() - prep_start_time) * 1000
            self.get_logger().info(f"Embedding preparation took: {prep_time:.2f}ms")
        
        # Process identities in batch
        if face_embeddings:
            if self.enable_debug_prints:
                identity_start_time = time.time()
            
            identity_results = self.identity_manager.process_new_embedding_batch(face_embeddings)
            
            if self.enable_debug_prints:
                identity_time = (time.time() - identity_start_time) * 1000
                self.get_logger().info(f"Identity processing took: {identity_time:.2f}ms for {len(face_embeddings)} faces")
            
            # Publish results for each valid face
            if self.enable_debug_prints:
                publish_start_time = time.time()
            
            for i in valid_indices:
                face_id = face_ids[i]
                landmarks_msg = landmarks_msgs[i]
                unique_id, confidence = identity_results.get(face_id, (None, 0.0))
                
                if self.enable_debug_prints:
                    self.get_logger().debug(f"Face {face_id} -> Identity: {unique_id}, Confidence: {confidence:.3f}")
                
                self._publish_recognition_result(landmarks_msg, unique_id, confidence)
            
            # Publish annotated image with all recognitions if enabled
            if self.enable_image_output and self.image_output_publisher and valid_indices:
                recognition_results = [(landmarks_msgs[i], identity_results.get(face_ids[i], (None, 0.0))) for i in valid_indices]
                self._publish_batch_annotated_image(recognition_results)
            
            if self.enable_debug_prints:
                publish_time = (time.time() - publish_start_time) * 1000
                self.get_logger().info(f"Publishing results took: {publish_time:.2f}ms for {len(valid_indices)} faces")
        else:
            self.get_logger().warning("No valid embeddings extracted for identity processing")


    
    def _extract_face_crop_from_landmarks(self, msg) -> Optional[np.ndarray]:
        """Extract face crop from landmarks message using bounding box."""
        try:
            if self.enable_debug_prints:
                self.get_logger().debug(f"Extracting face crop for face_id: {msg.face_id}")
                self.get_logger().debug(f"Bbox confidence: {msg.bbox_confidence}")
                self.get_logger().debug(f"Bbox xyxy: {msg.bbox_xyxy}")
                self.get_logger().debug(f"Bbox centroid: {msg.bbox_centroid}")
                
            # Use face bounding box from bbox_xyxy if available
            if len(msg.bbox_xyxy) >= 4 and msg.bbox_confidence > 0:
                # bbox_xyxy format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = msg.bbox_xyxy[:4]
                
                # Convert to pixel coordinates (assuming they are already in pixels)
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                if self.enable_debug_prints:
                    self.get_logger().debug(f"Using bbox_xyxy: x={x}, y={y}, w={w}, h={h}")
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, self.last_image.shape[1] - x)
                h = min(h, self.last_image.shape[0] - y)
                
                if w > 0 and h > 0:
                    face_crop = self.last_image[y:y+h, x:x+w]
                    if self.enable_debug_prints:
                        self.get_logger().debug(f"Extracted face crop from bbox: {face_crop.shape}")
                    return face_crop
                else:
                    if self.enable_debug_prints:
                        self.get_logger().debug(f"Invalid bbox dimensions after bounds check: w={w}, h={h}")
            
            # Fallback: estimate bounding box from landmarks
            landmarks = []
            valid_landmarks = 0
            for landmark in msg.landmarks:
                if landmark.c > 0:  # Valid landmark
                    # Convert normalized coordinates to pixel coordinates
                    x_pixel = int(landmark.x * msg.width)
                    y_pixel = int(landmark.y * msg.height)
                    landmarks.append([x_pixel, y_pixel])
                    valid_landmarks += 1
            
            if self.enable_debug_prints:
                self.get_logger().debug(f"Valid landmarks found: {valid_landmarks}")
            
            if len(landmarks) >= 4:  # Need at least 4 landmarks
                landmarks = np.array(landmarks)
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                
                if self.enable_debug_prints:
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
                
                if self.enable_debug_prints:
                    self.get_logger().debug(f"Landmark-based crop: x={x}, y={y}, w={w}, h={h}")
                
                if w > 0 and h > 0:
                    face_crop = self.last_image[y:y+h, x:x+w]
                    if self.enable_debug_prints:
                        self.get_logger().debug(f"Extracted face crop from landmarks: {face_crop.shape}")
                    return face_crop
                else:
                    if self.enable_debug_prints:
                        self.get_logger().debug(f"Invalid landmark-based dimensions: w={w}, h={h}")
            else:
                if self.enable_debug_prints:
                    self.get_logger().debug(f"Not enough valid landmarks: {len(landmarks)} (need at least 4)")
            
            self.get_logger().warning(f"Failed to extract face crop for face {msg.face_id}: no valid bbox or landmarks")
            return None
            
        except Exception as e:
            self.get_logger().error(f"Failed to extract face crop: {e}")
            return None
    

    
    def _publish_recognition_result(self, landmarks_msg, unique_id: Optional[str], confidence: float):
        """Publish facial recognition result."""
        try:
            # Create FacialRecognition message
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
            
            # Publish the message
            self.recognition_publisher.publish(recognition_msg)
            
            if unique_id:
                self.get_logger().debug(f"Published recognition: {landmarks_msg.face_id} -> {unique_id} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish recognition result: {e}")
    

    def _publish_batch_annotated_image(self, recognition_results: List):
        """Publish annotated image with recognition results for multiple faces."""
        if not self.enable_image_output or not self.image_output_publisher or self.last_image is None:
            return
        
        try:
            # Create a copy of the image for annotation
            annotated_image = self.last_image.copy()
            
            # Annotate all faces with recognition results
            for landmarks_msg, (unique_id, confidence) in recognition_results:
                self._draw_recognition_annotation(annotated_image, landmarks_msg, unique_id, confidence)
            
            # Convert to ROS image message
            image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            image_msg.header = self.last_image_header if self.last_image_header else recognition_results[0][0].header
            
            # Publish annotated image
            self.image_output_publisher.publish(image_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish batch annotated image: {e}")
    
    def _draw_recognition_annotation(self, image: np.ndarray, landmarks_msg, unique_id: Optional[str], confidence: float):
        """Draw recognition annotation on the image."""
        try:
            # Get face bounding box from bbox_xyxy
            if len(landmarks_msg.bbox_xyxy) >= 4 and landmarks_msg.bbox_confidence > 0:
                # bbox_xyxy format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = landmarks_msg.bbox_xyxy[:4]
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                # Draw face bounding box
                color = (0, 255, 0) if unique_id and unique_id != "unknown" else (0, 0, 255)  # Green for recognized, red for unknown
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Prepare text
                if unique_id and unique_id != "unknown":
                    text = f"{unique_id} ({confidence:.2f})"
                else:
                    text = "Unknown"
                
                # Draw text background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Background rectangle for text
                cv2.rectangle(image, (x, y - text_height - 10), (x + text_width + 10, y), color, -1)
                
                # Draw text
                cv2.putText(image, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)
                
                # Draw face ID (original detection ID)
                face_id_text = f"Face: {landmarks_msg.face_id}"
                cv2.putText(image, face_id_text, (x, y + h + 20), font, 0.4, (255, 255, 255), 1)
            
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
        # Save identity database before shutdown
        if self.identity_manager and hasattr(self.identity_manager, 'save_identity_database'):
            try:
                self.identity_manager.save_identity_database()
                self.get_logger().info("Identity database saved")
            except Exception as e:
                self.get_logger().error(f"Failed to save identity database: {e}")
        
        super().destroy_node()


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = FaceRecognitionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in face recognition node: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
