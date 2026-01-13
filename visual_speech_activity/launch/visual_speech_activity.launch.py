#!/usr/bin/env python3
"""
Launch file for visual speech activity detection node.
"""
import os
import subprocess
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")  # set AI_VENV or uses default


def _venv_site_packages(venv_path: str) -> str:
    """Get site-packages path from virtual environment."""
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True
    ).strip()


def _setup_visual_speech_activity(context, *args, **kwargs):
    """Setup visual speech activity node with virtual environment."""
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    # Get config file
    config_dir = get_package_share_directory("visual_speech_activity")
    config_file = os.path.join(config_dir, "config", "visual_speech_activity_params.yaml")

    # Load defaults from YAML
    with open(config_file, 'r') as f:
        params_yaml = yaml.safe_load(f)
        
    defaults = params_yaml['visual_speech_activity_node']['ros__parameters']

    # Visual speech activity node - use config file only, no parameter overrides
    # Launch arguments will override config file values when provided
    visual_speech_activity_node = Node(
        package='visual_speech_activity',
        executable='visual_speech_activity_node',
        name='visual_speech_activity_node',
        # Only pass config file - launch arguments will override automatically
        parameters=[config_file],
        output='screen',
        emulate_tty=True,
        additional_env={'PYTHONPATH': new_py_path}
    )

    return [
        LogInfo(msg=f"Using PYTHONPATH: {new_py_path}"),
        visual_speech_activity_node
    ]


def generate_launch_description():
    """Generate launch description for visual speech activity detection."""
    
    # Declare launch arguments
    recognition_input_topic_arg = DeclareLaunchArgument(
        'recognition_input_topic',
        default_value='/humans/faces/recognized',
        description='Input topic for facial recognition messages'
    )
    
    landmarks_input_topic_arg = DeclareLaunchArgument(
        'landmarks_input_topic',
        default_value='/humans/faces/detected',
        description='Input topic for facial landmarks messages'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/humans/faces/speaking',
        description='Output topic for speaking detection results'
    )
    
    ros4hri_with_id_arg = DeclareLaunchArgument(
        'ros4hri_with_id',
        default_value='true',
        description='Enable ROS4HRI per-ID mode (true) or array mode (false)'
    )
    
    window_size_arg = DeclareLaunchArgument(
        'window_size',
        default_value='20',
        description='Number of frames for temporal analysis window'
    )
    
    movement_threshold_arg = DeclareLaunchArgument(
        'movement_threshold',
        default_value='0.02',
        description='Minimum mouth aspect ratio variation to detect movement'
    )
    
    speaking_threshold_arg = DeclareLaunchArgument(
        'speaking_threshold',
        default_value='0.5',
        description='Confidence threshold for speaking classification'
    )
    
    temporal_smoothing_arg = DeclareLaunchArgument(
        'temporal_smoothing',
        default_value='true',
        description='Enable temporal smoothing of speaking detection'
    )
    
    min_frames_for_detection_arg = DeclareLaunchArgument(
        'min_frames_for_detection',
        default_value='5',
        description='Minimum frames required before speaking detection'
    )
    
    enable_debug_output_arg = DeclareLaunchArgument(
        'enable_debug_output',
        default_value='false',
        description='Enable debug output logging'
    )
    
    use_full_landmarks_arg = DeclareLaunchArgument(
        'use_full_landmarks',
        default_value='true',
        description='Use full 68-point dlib landmarks when available'
    )
    
    rnn_enabled_arg = DeclareLaunchArgument(
        'rnn_enabled',
        default_value='true',
        description='Enable RNN-based temporal classification'
    )
    
    use_face_recognition_arg = DeclareLaunchArgument(
        'use_face_recognition',
        default_value='true',
        description='Use face recognition for robust identity tracking'
    )

    return LaunchDescription([
        recognition_input_topic_arg,
        landmarks_input_topic_arg,
        output_topic_arg,
        ros4hri_with_id_arg,
        window_size_arg,
        movement_threshold_arg,
        speaking_threshold_arg,
        temporal_smoothing_arg,
        min_frames_for_detection_arg,
        enable_debug_output_arg,
        use_full_landmarks_arg,
        rnn_enabled_arg,
        use_face_recognition_arg,
        OpaqueFunction(function=_setup_visual_speech_activity)
    ])
