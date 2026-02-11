import pytest
import numpy as np
import rclpy
from face_detection.face_detector import FaceDetectorNode


class TestFaceDetectorNode:
    """Test suite for FaceDetectorNode basic functionality"""
    
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
        node = FaceDetectorNode()
        
        assert node.get_name() == "face_detector"
        assert node.bridge is not None
        assert node.frame_count == 0
        
        node.destroy_node()
    
    def test_node_has_required_attributes(self):
        """Test that the node has required attributes after initialization"""
        node = FaceDetectorNode()
        
        # Check for essential attributes
        assert hasattr(node, 'bridge')
        assert hasattr(node, 'detector')
        assert hasattr(node, 'frame_count')
        assert hasattr(node, 'total_processing_time')
        
        node.destroy_node()
    
    def test_timing_statistics_initialized(self):
        """Test that timing statistics are properly initialized"""
        node = FaceDetectorNode()
        
        assert node.frame_count == 0
        assert node.total_processing_time == 0.0
        assert node.max_processing_time == 0.0
        assert node.min_processing_time == float('inf')
        
        node.destroy_node()
    
    def test_node_parameters_exist(self):
        """Test that the node has declared parameters"""
        node = FaceDetectorNode()
        
        # Check that parameters exist by listing all parameters
        param_list = node.list_parameters([], depth=0)
        
        # Basic ROS parameters should exist (use_sim_time is always present)
        assert len(param_list.names) > 0
        
        node.destroy_node()
    
    def test_bridge_initialization(self):
        """Test that CV bridge is properly initialized"""
        node = FaceDetectorNode()
        
        # Verify bridge is initialized and can handle basic operations
        assert node.bridge is not None
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # The bridge should be able to convert basic numpy arrays
        # (This doesn't do full conversion, just checks the bridge exists)
        assert callable(getattr(node.bridge, 'cv2_to_imgmsg', None))
        assert callable(getattr(node.bridge, 'imgmsg_to_cv2', None))
        
        node.destroy_node()
