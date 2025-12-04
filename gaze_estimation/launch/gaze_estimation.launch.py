#!/usr/bin/env python3

import os
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")  # set AI_VENV or uses default

def _venv_site_packages(venv_path: str) -> str:
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True
    ).strip()


def _setup_gaze_estimation(context, *args, **kwargs):
    """Setup gaze estimation node with virtual environment."""
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    # Gaze estimation node
    gaze_estimation_node = Node(
        package='gaze_estimation',
        executable='gaze_estimation_node',
        name='gaze_estimation_node',
        parameters=[
            # LaunchConfiguration('config_file'),
            {
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'processing_rate_hz': LaunchConfiguration('processing_rate_hz'),
                'focal_length': LaunchConfiguration('focal_length'),
                'center_x_ratio': LaunchConfiguration('center_x_ratio'),
                'center_y_ratio': LaunchConfiguration('center_y_ratio'),
                'receiver_id': LaunchConfiguration('receiver_id'),
                'enable_image_output': LaunchConfiguration('enable_image_output'),
                'image_input_topic': LaunchConfiguration('image_input_topic'),
                'output_image_topic': LaunchConfiguration('output_image_topic'),
                'enable_debug_output': LaunchConfiguration('enable_debug_output'),
                'compressed_topic': LaunchConfiguration('compressed_topic'),
                'ros4hri_with_id': LaunchConfiguration('ros4hri_with_id'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )

    return [
        LogInfo(msg=f"[gaze_estimation] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[gaze_estimation] Injecting site-packages: {site_pkgs}"),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        gaze_estimation_node,
    ]


def generate_launch_description():
    # Get the launch directory
    pkg_gaze_estimation = get_package_share_directory('gaze_estimation')
    
    # Declare launch arguments
    # config_file_arg = DeclareLaunchArgument(
    #     'config_file',
    #     default_value=PathJoinSubstitution([
    #         FindPackageShare('gaze_estimation'),
    #         'config',
    #         'gaze_estimation.yaml'
    #     ]),
    #     description='Path to the config file'
    # )
    
    compressed_topic_arg = DeclareLaunchArgument(
        'compressed_topic',
        default_value='',
        description='Compressed image topic (if provided, uses compressed images instead of regular images)'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/humans/faces/detected',
        description='Input topic for facial landmarks'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic', 
        default_value='/humans/faces/gaze',
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
    
    output_image_topic_arg = DeclareLaunchArgument(
        'output_image_topic',
        default_value='/humans/faces/gaze/annotated_img',
        description='Output image topic with gaze visualization'
    )
    
    processing_rate_hz_arg = DeclareLaunchArgument(
        'processing_rate_hz',
        default_value='30.0',
        description='Processing rate in Hz'
    )
    
        
    enable_debug_output_arg = DeclareLaunchArgument(
        'enable_debug_output',
        default_value='false',
        description='Enable debug output'
    )
    
    ros4hri_with_id_arg = DeclareLaunchArgument(
        'ros4hri_with_id',
        default_value='false',
        description='Enable ROS4HRI with ID mode: subscribe to individual FacialLandmarks messages and publish individual Gaze messages per ID (default: ROS4HRI array mode)'
    )
    
    return LaunchDescription([
        # config_file_arg,
        compressed_topic_arg,
        input_topic_arg,
        output_topic_arg,
        enable_debug_output_arg,
        focal_length_arg,
        center_x_ratio_arg,
        center_y_ratio_arg,
        receiver_id_arg,
        enable_image_output_arg,
        image_input_topic_arg,
        output_image_topic_arg,
        processing_rate_hz_arg,
        ros4hri_with_id_arg,
        OpaqueFunction(function=_setup_gaze_estimation),
    ])
