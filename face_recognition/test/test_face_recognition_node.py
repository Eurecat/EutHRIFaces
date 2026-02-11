import pytest
import numpy as np
import rclpy
from face_recognition.face_recognition_node import FaceRecognitionNode


class TestFaceRecognitionNode:
    """Test suite for FaceRecognitionNode basic functionality"""
    
    @classmethod
    def setup_class(cls):
        """Initialize ROS once for all tests in this class"""
        rclpy.init()
    
    @classmethod
    def teardown_class(cls):
        """Shutdown ROS after all tests"""
        rclpy.shutdown()
    
    def test_node_initialization(self):
        """Test that the node initializes correctly"""
        node = FaceRecognitionNode()
        
        assert node.get_name() == "face_recognition_node"
        assert node.cv_bridge is not None
        
        node.destroy_node()
    
    def test_node_has_required_attributes(self):
        """Test that the node has required attributes after initialization"""
        node = FaceRecognitionNode()
        
        # Check for essential attributes
        assert hasattr(node, 'cv_bridge')
        assert hasattr(node, 'face_embedding_extractor')
        assert hasattr(node, 'identity_manager')
        assert hasattr(node, 'total_processing_time')
        assert hasattr(node, 'processed_messages')
        
        node.destroy_node()
    
    def test_performance_tracking_initialized(self):
        """Test that performance tracking variables are properly initialized"""
        node = FaceRecognitionNode()
        
        assert node.total_processing_time == 0.0
        assert node.processed_messages == 0
        
        node.destroy_node()
    
    def test_qos_profile_exists(self):
        """Test that QoS profile is configured"""
        node = FaceRecognitionNode()
        
        assert hasattr(node, 'qos_profile')
        assert node.qos_profile is not None
        
        node.destroy_node()
    
    def test_cv_bridge_initialization(self):
        """Test that CV bridge is properly initialized"""
        node = FaceRecognitionNode()
        
        assert node.cv_bridge is not None
        
        # Verify bridge methods exist
        assert callable(getattr(node.cv_bridge, 'cv2_to_imgmsg', None))
        assert callable(getattr(node.cv_bridge, 'imgmsg_to_cv2', None))
        
        node.destroy_node()
