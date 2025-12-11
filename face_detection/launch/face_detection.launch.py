#!/usr/bin/env python3
"""
Launch file for face detection node.
"""
import os
import subprocess
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")  # set AI_VENV or uses default

def _venv_site_packages(venv_path: str) -> str:
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True
    ).strip()


def _setup_face_detection(context, *args, **kwargs):
    """Setup face detection node with virtual environment."""
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    # Get config file
    config_dir = get_package_share_directory("face_detection")
    config_file = os.path.join(config_dir, "config", "face_detection_params.yaml")

    # Load defaults from YAML
    with open(config_file, 'r') as f:
        params_yaml = yaml.safe_load(f)
        
    defaults = params_yaml['face_detector']['ros__parameters']

    # Build node_params dict with LaunchConfiguration overrides
    node_params = {}
    params_to_expose = [
        'input_topic',
        'output_topic',
        'output_image_topic',
        'compressed_topic',
        'processing_rate_hz',
        'device',
        'model_path',
        'confidence_threshold',
        'iou_threshold',
        'enable_debug_output',
        'face_id_prefix',
        'enable_image_output',
        'face_bbox_thickness',
        'face_landmark_radius',
        'face_bbox_color',
        'face_landmark_color',
        'use_boxmot',
        'boxmot_tracker_type',
        'boxmot_reid_model',
        'ros4hri_with_id'
    ]

    for param in params_to_expose:
        if param in defaults:
            node_params[param] = LaunchConfiguration(param)

    # Face detection node
    face_detection_node = Node(
        package='face_detection',
        executable='face_detector',
        name='face_detector',
        # Pass config file first, then overrides
        parameters=[config_file, node_params],
        output='screen',
        emulate_tty=True,
    )

    return [
        LogInfo(msg=f"[face_detection] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[face_detection] Injecting site-packages: {site_pkgs}"),
        LogInfo(msg=f"[face_detection] Loading config from: {config_file}"),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        face_detection_node,
    ]


def generate_launch_description():
    """Generate launch description for face detection."""
    
    # Get config file
    config_dir = get_package_share_directory("face_detection")
    config_file = os.path.join(config_dir, "config", "face_detection_params.yaml")

    # Load defaults from YAML
    with open(config_file, 'r') as f:
        params_yaml = yaml.safe_load(f)
        
    defaults = params_yaml['face_detector']['ros__parameters']

    # Declare launch arguments
    launch_args = []
    params_to_expose = [
        'input_topic',
        'output_topic',
        'output_image_topic',
        'compressed_topic',
        'processing_rate_hz',
        'device',
        'model_path',
        'confidence_threshold',
        'iou_threshold',
        'enable_debug_output',
        'face_id_prefix',
        'enable_image_output',
        'face_bbox_thickness',
        'face_landmark_radius',
        'face_bbox_color',
        'face_landmark_color',
        'use_boxmot',
        'boxmot_tracker_type',
        'boxmot_reid_model',
        'ros4hri_with_id'
    ]

    for param in params_to_expose:
        if param in defaults:
            launch_args.append(
                DeclareLaunchArgument(
                    param,
                    default_value=str(defaults[param]),
                    description=f'Parameter {param} from face_detection_params.yaml'
                )
            )
    
    return LaunchDescription(
        launch_args +
        [
            OpaqueFunction(function=_setup_face_detection),
        ]
    )
