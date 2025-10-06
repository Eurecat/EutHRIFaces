#!/usr/bin/env python3
"""
Launch file for face detection node.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for face detection."""
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('face_detection'),
            'config',
            'face_detection.yaml'
        ]),
        description='Path to the face detection configuration file'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/color/image_rect_raw',
        description='Input image topic'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic', 
        default_value='/people/faces/detected',
        description='Output facial landmarks topic'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Device to run inference on (cpu/cuda)'
    )
    
    enable_debug_output_arg = DeclareLaunchArgument(
        'enable_debug_output',
        default_value='false',
        description='Enable debug output'
    )
    
    # YOLO Face Detection Parameters
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='weights/yolov8n-face.onnx',
        description='Path to YOLO face detection model'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.1',
        description='Confidence threshold for face detection'
    )
    
    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold',
        default_value='0.4',
        description='IoU threshold for non-maximum suppression'
    )
    
    # General parameters
    face_id_prefix_arg = DeclareLaunchArgument(
        'face_id_prefix',
        default_value='face_',
        description='Prefix for face IDs'
    )
    
    # BOXMOT tracking parameters
    use_boxmot_arg = DeclareLaunchArgument(
        'use_boxmot',
        default_value='true',
        description='Enable BOXMOT tracking for face detection'
    )
    
    boxmot_tracker_type_arg = DeclareLaunchArgument(
        'boxmot_tracker_type',
        default_value='bytetrack',
        description='Type of BOXMOT tracker (bytetrack, botsort, strongsort, etc.)'
    )
    
    boxmot_reid_model_arg = DeclareLaunchArgument(
        'boxmot_reid_model',
        default_value='',
        description='Path to ReID model for BOXMOT tracking'
    )
    
    # Image visualization parameters  
    output_image_topic_arg = DeclareLaunchArgument(
        'output_image_topic',
        default_value='/people/faces/detected/image_with_faces',
        description='Output topic for visualization images'
    )
    
    enable_image_output_arg = DeclareLaunchArgument(
        'enable_image_output',
        default_value='true',
        description='Enable image visualization output'
    )
    
    face_bbox_thickness_arg = DeclareLaunchArgument(
        'face_bbox_thickness',
        default_value='2',
        description='Thickness of face bounding box lines'
    )
    
    face_landmark_radius_arg = DeclareLaunchArgument(
        'face_landmark_radius',
        default_value='3',
        description='Radius of facial landmark circles'
    )
    
    face_bbox_color_arg = DeclareLaunchArgument(
        'face_bbox_color',
        default_value='[0, 255, 0]',
        description='Color of face bounding boxes (BGR format)'
    )
    
    face_landmark_color_arg = DeclareLaunchArgument(
        'face_landmark_color',
        default_value='[255, 0, 0]',
        description='Color of facial landmarks (BGR format)'
    )
    
    # Face detection node
    face_detection_node = Node(
        package='face_detection',
        executable='face_detector',
        name='face_detector',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'output_image_topic': LaunchConfiguration('output_image_topic'),
                'device': LaunchConfiguration('device'),
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'iou_threshold': LaunchConfiguration('iou_threshold'),
                'enable_debug_output': LaunchConfiguration('enable_debug_output'),
                'face_id_prefix': LaunchConfiguration('face_id_prefix'),
                'enable_image_output': LaunchConfiguration('enable_image_output'),
                'face_bbox_thickness': LaunchConfiguration('face_bbox_thickness'),
                'face_landmark_radius': LaunchConfiguration('face_landmark_radius'),
                'face_bbox_color': LaunchConfiguration('face_bbox_color'),
                'face_landmark_color': LaunchConfiguration('face_landmark_color'),
                'use_boxmot': LaunchConfiguration('use_boxmot'),
                'boxmot_tracker_type': LaunchConfiguration('boxmot_tracker_type'),
                'boxmot_reid_model': LaunchConfiguration('boxmot_reid_model'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        config_file_arg,
        input_topic_arg,
        output_topic_arg,
        output_image_topic_arg,
        device_arg,
        model_path_arg,
        confidence_threshold_arg,
        iou_threshold_arg,
        enable_debug_output_arg,
        face_id_prefix_arg,
        enable_image_output_arg,
        face_bbox_thickness_arg,
        face_landmark_radius_arg,
        face_bbox_color_arg,
        face_landmark_color_arg,
        use_boxmot_arg,
        boxmot_tracker_type_arg,
        boxmot_reid_model_arg,
        face_detection_node,
    ])
