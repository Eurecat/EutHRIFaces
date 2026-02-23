#!/usr/bin/env python3
"""
Launch file for face recognition node.

This launch file starts the face recognition node with configurable parameters
for face embedding extraction and identity management based on the EUT YOLO approach.
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


def _setup_face_recognition(context, *args, **kwargs):
    """Setup face recognition node with virtual environment."""
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    # Get config file
    config_dir = get_package_share_directory("face_recognition")
    config_file = os.path.join(config_dir, "config", "face_recognition_params.yaml")

    # Load defaults from YAML
    with open(config_file, 'r') as f:
        params_yaml = yaml.safe_load(f)
        
    defaults = params_yaml['face_recognition_node']['ros__parameters']

    # Check if debug output is enabled
    enable_debug = LaunchConfiguration('enable_debug_output').perform(context).lower() == 'true'
    
    # Prepare arguments - add debug log level if debug output is enabled
    node_arguments = []
    if enable_debug:
        # Set debug level only for this specific node, not globally
        node_arguments = ['--ros-args', '--log-level', 'face_recognition_node:=debug']

    # Build node_params dict with LaunchConfiguration overrides
    node_params = {}
    params_to_expose = [
        'input_topic',
        'output_topic',
        'image_input_topic',
        'compressed_topic',
        'processing_rate_hz',
        'device',
        'face_embedding_model',
        'face_embedding_weights_name',
        'similarity_threshold',
        'clustering_threshold',
        'max_embeddings_per_identity',
        'identity_timeout',
        'identity_database_path',
        'enable_debug_output',
        'receiver_id',
        'ros4hri_with_id',
        'min_h_size'
    ]

    for param in params_to_expose:
        if param in defaults:
            node_params[param] = LaunchConfiguration(param)

    # Face recognition node
    face_recognition_node = Node(
        package='face_recognition',
        executable='face_recognition_node',
        name='face_recognition_node',
        # Pass config file first, then overrides
        parameters=[config_file, node_params],
        arguments=node_arguments,
        output='screen',
        emulate_tty=True,
    )

    return [
        LogInfo(msg=f"[face_recognition] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[face_recognition] Injecting site-packages: {site_pkgs}"),
        LogInfo(msg=f"[face_recognition] Loading config from: {config_file}"),
        LogInfo(msg=f"[face_recognition] Debug logging: {'enabled' if enable_debug else 'disabled'}"),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        face_recognition_node,
    ]


def generate_launch_description():
    """Generate launch description for face recognition."""
    
    # Get config file
    config_dir = get_package_share_directory("face_recognition")
    config_file = os.path.join(config_dir, "config", "face_recognition_params.yaml")

    # Load defaults from YAML
    with open(config_file, 'r') as f:
        params_yaml = yaml.safe_load(f)
        
    defaults = params_yaml['face_recognition_node']['ros__parameters']

    # Declare launch arguments
    launch_args = []
    params_to_expose = [
        'input_topic',
        'output_topic',
        'image_input_topic',
        'compressed_topic',
        'processing_rate_hz',
        'device',
        'face_embedding_model',
        'face_embedding_weights_name',
        'similarity_threshold',
        'clustering_threshold',
        'max_embeddings_per_identity',
        'identity_timeout',
        'identity_database_path',
        'enable_debug_output',
        'receiver_id',
        'ros4hri_with_id',
        'min_h_size'
    ]

    for param in params_to_expose:
        if param in defaults:
            launch_args.append(
                DeclareLaunchArgument(
                    param,
                    default_value=str(defaults[param]),
                    description=f'Parameter {param} from face_recognition_params.yaml'
                )
            )
    
    return LaunchDescription(
        launch_args +
        [
            OpaqueFunction(function=_setup_face_recognition),
        ]
    )
