#!/usr/bin/env python3
"""
Launch file for face recognition node.

This launch file starts the face recognition node with configurable parameters
for face embedding extraction and identity management based on the EUT YOLO approach.
"""

import os
import subprocess
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

    # Face recognition node
    face_recognition_node = Node(
        package='face_recognition',
        executable='face_recognition_node',
        name='face_recognition_node',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'image_input_topic': LaunchConfiguration('image_input_topic'),
                'processing_rate_hz': LaunchConfiguration('processing_rate_hz'),
                'device': LaunchConfiguration('device'),
                'face_embedding_model': LaunchConfiguration('face_embedding_model'),
                'weights_path': LaunchConfiguration('weights_path'),
                'face_embedding_weights_name': LaunchConfiguration('face_embedding_weights_name'),
                'similarity_threshold': LaunchConfiguration('similarity_threshold'),
                'clustering_threshold': LaunchConfiguration('clustering_threshold'),
                'max_embeddings_per_identity': LaunchConfiguration('max_embeddings_per_identity'),
                'identity_timeout': LaunchConfiguration('identity_timeout'),
                'identity_database_path': LaunchConfiguration('identity_database_path'),
                'enable_debug_prints': LaunchConfiguration('enable_debug_prints'),
                'receiver_id': LaunchConfiguration('receiver_id'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )

    return [
        LogInfo(msg=f"[face_recognition] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[face_recognition] Injecting site-packages: {site_pkgs}"),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        face_recognition_node,
    ]


def generate_launch_description():
    """Generate launch description for face recognition."""
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('face_recognition'),
            'config',
            'face_recognition.yaml'
        ]),
        description='Path to the face recognition configuration file'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/humans/faces/detected',
        description='Input topic for facial landmarks'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic', 
        default_value='/humans/faces/recognized',
        description='Output topic for facial recognition results'
    )
    
    image_input_topic_arg = DeclareLaunchArgument(
        'image_input_topic',
        default_value='/camera/color/image_rect_raw',
        description='Input RGB image topic'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda:0',
        description='Device to run inference on (cpu/cuda)'
    )
    
    # Face embedding parameters
    face_embedding_model_arg = DeclareLaunchArgument(
        'face_embedding_model',
        default_value='vggface2',
        description='Face embedding model to use (vggface2, casia-webface)'
    )
    
    weights_path_arg = DeclareLaunchArgument(
        'weights_path',
        default_value='weights',
        description='Path to weights directory (relative to package root)'
    )
    
    face_embedding_weights_name_arg = DeclareLaunchArgument(
        'face_embedding_weights_name',
        default_value='20180402-114759-vggface2.pt',
        description='Specific filename of face embedding weights'
    )
    
    # Identity management parameters
    similarity_threshold_arg = DeclareLaunchArgument(
        'similarity_threshold',
        default_value='0.25',
        description='Minimum similarity threshold for identity assignment'
    )
    
    clustering_threshold_arg = DeclareLaunchArgument(
        'clustering_threshold',
        default_value='0.3',
        description='Threshold for clustering embeddings into identities'
    )
    
    max_embeddings_per_identity_arg = DeclareLaunchArgument(
        'max_embeddings_per_identity',
        default_value='250',
        description='Maximum embeddings to store per identity'
    )
    
    identity_timeout_arg = DeclareLaunchArgument(
        'identity_timeout',
        default_value='60.0',
        description='Time (seconds) after which inactive identity is removed'
    )
    
    identity_database_path_arg = DeclareLaunchArgument(
        'identity_database_path',
        default_value='',
        description='Path to persistent identity database JSON file'
    )
    enable_debug_prints_arg = DeclareLaunchArgument(
        'enable_debug_prints',
        default_value='true',
        description='Enable detailed debug output'
    )

    receiver_id_arg = DeclareLaunchArgument(
        'receiver_id',
        default_value='face_recognition',
        description='Receiver ID for hri_msgs'
    )
    
    processing_rate_hz_arg = DeclareLaunchArgument(
        'processing_rate_hz',
        default_value='30.0',
        description='Processing rate in Hz'
    )
    
    return LaunchDescription([
        # Launch arguments
        config_file_arg,
        input_topic_arg,
        output_topic_arg,
        image_input_topic_arg,
        device_arg,
        face_embedding_model_arg,
        weights_path_arg,
        face_embedding_weights_name_arg,
        similarity_threshold_arg,
        clustering_threshold_arg,
        max_embeddings_per_identity_arg,
        identity_timeout_arg,
        identity_database_path_arg,
        enable_debug_prints_arg,
        receiver_id_arg,
        processing_rate_hz_arg,
        
        # Node with virtual environment setup
        OpaqueFunction(function=_setup_face_recognition),
    ])
