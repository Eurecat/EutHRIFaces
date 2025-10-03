#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the launch directory
    pkg_gaze_estimation = get_package_share_directory('gaze_estimation')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('gaze_estimation'),
            'config',
            'gaze_estimation.yaml'
        ]),
        description='Path to the config file'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/face_detection/facial_landmarks',
        description='Input topic for facial landmarks'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic', 
        default_value='/gaze_estimation/gaze',
        description='Output topic for gaze messages'
    )
    
    # Camera parameters
    focal_length_arg = DeclareLaunchArgument(
        'focal_length',
        default_value='640.0',
        description='Camera focal length (defaults to image width if not specified)'
    )
    
    center_x_ratio_arg = DeclareLaunchArgument(
        'center_x_ratio',
        default_value='0.5',
        description='Camera center X as ratio of image width (0.5 = center)'
    )
    
    center_y_ratio_arg = DeclareLaunchArgument(
        'center_y_ratio',
        default_value='0.5', 
        description='Camera center Y as ratio of image height (0.5 = center)'
    )
    
    receiver_id_arg = DeclareLaunchArgument(
        'receiver_id',
        default_value='pin_hole_cam_model',
        description='Receiver ID for gaze messages'
    )
    
    # Image visualization parameters
    enable_image_output_arg = DeclareLaunchArgument(
        'enable_image_output',
        default_value='true',
        description='Enable gaze visualization image output'
    )
    
    image_input_topic_arg = DeclareLaunchArgument(
        'image_input_topic',
        default_value='/camera/color/image_rect_raw',
        description='Input image topic for visualization'
    )
    
    image_output_topic_arg = DeclareLaunchArgument(
        'image_output_topic',
        default_value='/gaze_estimation/image_with_gaze',
        description='Output image topic with gaze visualization'
    )
    
    # Gaze estimation node
    gaze_estimation_node = Node(
        package='gaze_estimation',
        executable='gaze_estimation_node',
        name='gaze_estimation_node',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'focal_length': LaunchConfiguration('focal_length'),
                'center_x_ratio': LaunchConfiguration('center_x_ratio'),
                'center_y_ratio': LaunchConfiguration('center_y_ratio'),
                'receiver_id': LaunchConfiguration('receiver_id'),
                'enable_image_output': LaunchConfiguration('enable_image_output'),
                'image_input_topic': LaunchConfiguration('image_input_topic'),
                'image_output_topic': LaunchConfiguration('image_output_topic'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        config_file_arg,
        input_topic_arg,
        output_topic_arg,
        focal_length_arg,
        center_x_ratio_arg,
        center_y_ratio_arg,
        receiver_id_arg,
        enable_image_output_arg,
        image_input_topic_arg,
        image_output_topic_arg,
        gaze_estimation_node,
    ])
