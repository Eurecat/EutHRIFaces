#!/usr/bin/env python3
"""
Launch file for visual speech activity detection node.
"""
import os
import subprocess
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo, SetEnvironmentVariable
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

    # Build node_params dict with LaunchConfiguration overrides
    node_params = {}
    params_to_expose = [
        'recognition_input_topic',
        'landmarks_input_topic',
        'output_topic',
        'output_image_topic',
        'ros4hri_with_id',
        'speaking_threshold',
        'vsdlm_weights_path',
        'vsdlm_weights_name',
        'vsdlm_model_variant',
        'vsdlm_execution_provider',
        'vsdlm_mouth_height_ratio',
        'vsdlm_temporal_smoothing',
        'vsdlm_smoothing_window_size',
        'vsdlm_min_confidence_for_change',
        'window_size',
        'movement_threshold',
        'min_frames_for_detection',
        'use_full_landmarks',
        'rnn_enabled',
        'enable_debug_output',
        'vsdlm_debug_save_crops',
        'image_topic',
        'compressed_topic',
        'use_face_recognition',
        'enable_image_output',
        'label_offset_y'
    ]

    for param in params_to_expose:
        if param in defaults:
            node_params[param] = LaunchConfiguration(param)

    # Visual speech activity node
    visual_speech_activity_node = Node(
        package='visual_speech_activity',
        executable='visual_speech_activity_node',
        name='visual_speech_activity_node',
        # Pass config file first, then overrides
        parameters=[config_file, node_params],
        output='screen',
        emulate_tty=True,
    )

    return [
        LogInfo(msg=f"[visual_speech_activity] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[visual_speech_activity] Injecting site-packages: {site_pkgs}"),
        LogInfo(msg=f"[visual_speech_activity] Loading config from: {config_file}"),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        visual_speech_activity_node,
    ]


def generate_launch_description():
    """Generate launch description for visual speech activity detection."""
    
    # Get config file
    config_dir = get_package_share_directory("visual_speech_activity")
    config_file = os.path.join(config_dir, "config", "visual_speech_activity_params.yaml")

    # Load defaults from YAML
    with open(config_file, 'r') as f:
        params_yaml = yaml.safe_load(f)
        
    defaults = params_yaml['visual_speech_activity_node']['ros__parameters']

    # Declare launch arguments
    launch_args = []
    params_to_expose = [
        'recognition_input_topic',
        'landmarks_input_topic',
        'output_topic',
        'output_image_topic',
        'ros4hri_with_id',
        'speaking_threshold',
        'vsdlm_weights_path',
        'vsdlm_weights_name',
        'vsdlm_model_variant',
        'vsdlm_execution_provider',
        'vsdlm_mouth_height_ratio',
        'vsdlm_temporal_smoothing',
        'vsdlm_smoothing_window_size',
        'vsdlm_min_confidence_for_change',
        'window_size',
        'movement_threshold',
        'min_frames_for_detection',
        'use_full_landmarks',
        'rnn_enabled',
        'enable_debug_output',
        'vsdlm_debug_save_crops',
        'image_topic',
        'compressed_topic',
        'use_face_recognition',
        'enable_image_output',
        'label_offset_y'
    ]

    for param in params_to_expose:
        if param in defaults:
            launch_args.append(
                DeclareLaunchArgument(
                    param,
                    default_value=str(defaults[param]),
                    description=f'Parameter {param} from visual_speech_activity_params.yaml'
                )
            )
    
    return LaunchDescription(
        launch_args +
        [
            OpaqueFunction(function=_setup_visual_speech_activity),
        ]
    )
