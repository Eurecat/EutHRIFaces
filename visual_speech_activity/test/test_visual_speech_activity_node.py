import pytest
import numpy as np
import rclpy
from visual_speech_activity.visual_speech_activity_node import VisualSpeechActivityNode


class TestVisualSpeechActivityNode:
    """Test suite for VisualSpeechActivityNode basic functionality"""
    
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
        node = VisualSpeechActivityNode()
        
        assert node.get_name() == "visual_speech_activity_node"
        assert node.vsdlm_detector is not None
        
        node.destroy_node()
    
    def test_node_has_required_attributes(self):
        """Test that the node has required attributes after initialization"""
        node = VisualSpeechActivityNode()
        
        # Check for essential attributes
        assert hasattr(node, 'vsdlm_detector')
        
        node.destroy_node()
    
    def test_vsdlm_detector_initialized(self):
        """Test that VSDLM detector is properly initialized"""
        node = VisualSpeechActivityNode()
        
        assert node.vsdlm_detector is not None
        
        node.destroy_node()
    
    def test_node_parameters_exist(self):
        """Test that the node has declared parameters"""
        node = VisualSpeechActivityNode()
        
        # Check that parameters exist
        param_names = [param.name for param in node.get_parameters([])]
        
        # Basic ROS parameters should exist
        assert len(param_names) > 0
        
        node.destroy_node()
    
    def test_stamp_to_float_conversion(self):
        """Test timestamp conversion utility function"""
        from visual_speech_activity.visual_speech_activity_node import _stamp_to_float
        from rclpy.time import Time
        
        # Create a test timestamp
        test_time = Time(seconds=10, nanoseconds=500000000)  # 10.5 seconds
        
        result = _stamp_to_float(test_time)
        
        # Should be approximately 10.5 seconds
        assert isinstance(result, float)
        assert 10.4 < result < 10.6  # Allow for floating point precision
