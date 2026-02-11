import pytest
import numpy as np
import rclpy
from gaze_estimation.gaze_estimation_node import GazeEstimationNode


class TestGazeEstimationNode:
    """Test suite for GazeEstimationNode basic functionality"""
    
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
        node = GazeEstimationNode()
        
        assert node.get_name() == "gaze_estimation_node"
        assert node.frame_count == 0
        
        node.destroy_node()
    
    def test_node_has_required_attributes(self):
        """Test that the node has required attributes after initialization"""
        node = GazeEstimationNode()
        
        # Check for essential attributes
        assert hasattr(node, 'frame_count')
        assert hasattr(node, 'total_processing_time')
        assert hasattr(node, 'qos_profile')
        
        node.destroy_node()
    
    def test_timing_statistics_initialized(self):
        """Test that timing statistics are properly initialized"""
        node = GazeEstimationNode()
        
        assert node.frame_count == 0
        assert node.total_processing_time == 0.0
        assert node.max_processing_time == 0.0
        assert node.min_processing_time == float('inf')
        
        node.destroy_node()
    
    def test_qos_profile_exists(self):
        """Test that QoS profile is configured"""
        node = GazeEstimationNode()
        
        assert hasattr(node, 'qos_profile')
        assert node.qos_profile is not None
        
        node.destroy_node()
    
    def test_node_parameters_exist(self):
        """Test that the node has declared parameters"""
        node = GazeEstimationNode()
        
        # Check that parameters exist by listing all parameters
        param_list = node.list_parameters([], depth=0)
        
        # Basic ROS parameters should exist (use_sim_time is always present)
        assert len(param_list.names) > 0
        
        node.destroy_node()
