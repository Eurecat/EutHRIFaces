import pytest
import numpy as np
import rclpy
from visual_speech_activity.visual_speech_activity_node import VisualSpeechActivityNode


def test_module_import():
    """Simple sanity check to verify module imports correctly"""
    assert VisualSpeechActivityNode is not None


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
        
        # Check that parameters exist by listing all parameters
        param_list = node.list_parameters([], depth=0)
        
        # Basic ROS parameters should exist (use_sim_time is always present)
        assert len(param_list.names) > 0
        
        node.destroy_node()
    
    def test_stamp_to_float_conversion(self):
        """Test timestamp conversion utility function"""
        from visual_speech_activity.visual_speech_activity_node import _stamp_to_float
        from builtin_interfaces.msg import Time
        
        # Create a test timestamp (builtin_interfaces.msg.Time)
        test_time = Time()
        test_time.sec = 10
        test_time.nanosec = 500000000  # 0.5 seconds
        
        result = _stamp_to_float(test_time)
        
        # Should be approximately 10.5 seconds
        assert isinstance(result, float)
        assert 10.4 < result < 10.6  # Allow for floating point precision
