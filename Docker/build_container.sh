#!/usr/bin/env bash
#
# Build script for EUT HRI Human Face Docker container
#
# Usage:
# - Default (Vulcanexus): ./build_container.sh
# - Standard ROS2: ./build_container.sh --standard-ros
# - CPU-only version: ./build_container.sh --cpu
# - CPU-only Vulcanexus: ./build_container.sh --cpu --vulcanexus
# - Clean rebuild: ./build_container.sh --clean-rebuild [--standard-ros] [--cpu]
#

export DOCKER_BUILDKIT=1

set -e

# --- BEGIN: Manage .env file ---
ENV_FILE="./.env" # Assuming .env is in the same directory as this script
# Create .env if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating $ENV_FILE..."
    touch "$ENV_FILE"
fi

# Define directories
DEPS_DIR="./deps"
MAIN_DIR=".."

# Create deps directory if it doesn't exist
if [ ! -d $DEPS_DIR ]; then
    mkdir -p $DEPS_DIR
fi

# Check arguments
TARGET_DISTRO="jazzy"
BASE_IMAGE="eut_ros_torch:${TARGET_DISTRO}"
CPU_ONLY="false"
REBUILD=false
NO_VCS=false
USE_VULCANEXUS=false
USE_HUMBLE=false
for arg in "$@"; do
    if [ "$arg" == "--clean-rebuild" ]; then
        REBUILD=true
    fi
    if [ "$arg" == "--vulcanexus" ]; then
        BASE_IMAGE="eut_ros_vulcanexus_torch:${TARGET_DISTRO}"
        USE_VULCANEXUS=true
    fi
    if [ "$arg" == "--cpu" ]; then
        CPU_ONLY="true"
    fi
    if [ "$arg" == "--humble" ]; then
        TARGET_DISTRO="humble"
        BASE_IMAGE="eut_ros_torch:${TARGET_DISTRO}"
        USE_HUMBLE=true
    fi
    if [ "$arg" == "--no-vcs" ]; then
        NO_VCS=true
    fi
done

# Validate that Vulcanexus and Humble are not used together
if $USE_VULCANEXUS && $USE_HUMBLE; then
    echo "ERROR: --vulcanexus and --humble cannot be used together."
    echo "Vulcanexus is only available for Jazzy."
    exit 1
fi

# Update base image for CPU variant
if [ "$CPU_ONLY" = "true" ]; then
    if [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
        BASE_IMAGE="eut_ros_vulcanexus_torch_cpu:${TARGET_DISTRO}"
    else
        BASE_IMAGE="eut_ros_torch_cpu:${TARGET_DISTRO}"
    fi
fi

# Validate that Vulcanexus and Humble are not used together
if $USE_VULCANEXUS && $USE_HUMBLE; then
    echo "ERROR: --vulcanexus and --humble cannot be used together."
    echo "Vulcanexus is only available for Jazzy."
    exit 1
fi

if $REBUILD; then
    echo "Rebuilding: cleaning up dependencies..."
    rm -rf $DEPS_DIR/*
fi

# Import/update dependencies repository using VCS tools (currently empty)
if ! $NO_VCS; then
    echo "Importing/updating dependencies repository using VCS..."
    if [ -s faces.repos ]; then
        vcs import ${DEPS_DIR} < faces.repos
        vcs pull ${DEPS_DIR}
    else
        echo "No external dependencies defined in faces.repos"
    fi
else
    echo "Skipping VCS operations..."
fi

# Set image name based on the base image choice and CPU flag
if [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
    echo "Building with Vulcanexus ${TARGET_DISTRO} base image..."
else
    echo "Building with standard ROS2 ${TARGET_DISTRO} base image..."
fi


# Set image name based on the base image choice and CPU flag
if [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
    if [ "$CPU_ONLY" = "true" ]; then
        IMAGE_NAME="eut_human_face_vulcanexus_cpu:${TARGET_DISTRO}"
    else
        IMAGE_NAME="eut_human_face_vulcanexus:${TARGET_DISTRO}"
    fi
else
    if [ "$CPU_ONLY" = "true" ]; then
        IMAGE_NAME="eut_human_face_cpu:${TARGET_DISTRO}"
    else
        IMAGE_NAME="eut_human_face:${TARGET_DISTRO}"
    fi
fi


echo "Base image: ${BASE_IMAGE}"
echo "CPU Only: ${CPU_ONLY}"
echo "Output image: ${IMAGE_NAME}"

if $REBUILD; then
    echo "Rebuilding the application Docker image..."
    docker build --no-cache . --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg CPU_ONLY="${CPU_ONLY}" -t ${IMAGE_NAME} -f Dockerfile
else
    docker build . --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg CPU_ONLY="${CPU_ONLY}" -t ${IMAGE_NAME} -f Dockerfile
fi

# Set or Update BUILT_IMAGE 
if grep -q -E "^BUILT_IMAGE=" "$ENV_FILE"; then
    sed -i "s/^BUILT_IMAGE=.*/BUILT_IMAGE=${IMAGE_NAME}/" "$ENV_FILE"
else
    echo "BUILT_IMAGE=${IMAGE_NAME}" >> "$ENV_FILE"
fi

# Set or Update DOCKER_RUNTIME based on CPU_ONLY flag
if [ "$CPU_ONLY" = "true" ]; then
    DOCKER_RUNTIME="runc"
else
    DOCKER_RUNTIME="nvidia"
fi

if grep -q -E "^DOCKER_RUNTIME=" "$ENV_FILE"; then
    sed -i "s/^DOCKER_RUNTIME=.*/DOCKER_RUNTIME=$DOCKER_RUNTIME/" "$ENV_FILE"
else
    echo "DOCKER_RUNTIME=$DOCKER_RUNTIME" >> "$ENV_FILE"
fi

# Set or Update TARGET_DISTRO
if grep -q -E "^TARGET_DISTRO=" "$ENV_FILE"; then
    sed -i "s/^TARGET_DISTRO=.*/TARGET_DISTRO=$TARGET_DISTRO/" "$ENV_FILE"
else
    echo "TARGET_DISTRO=$TARGET_DISTRO" >> "$ENV_FILE"
fi

# Set or Update RMW_IMPLEMENTATION based on TARGET_DISTRO
if [ "$TARGET_DISTRO" = "humble" ]; then
    RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
    IMG_RAW_TOPIC="/head_front_camera/color/image_raw/compressed"
    LD_LIBRARY_PATH=/workspace/install/hri_msgs/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/opt/ros_python_env/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/ros_python_env/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/ros_python_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:
else
    RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
    IMG_RAW_TOPIC="/camera/image_raw/compressed"
    LD_LIBRARY_PATH=/workspace/install/hri_msgs/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/lib:/opt/ros_python_env/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/ros_python_env/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/ros_python_env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:
fi

if grep -q -E "^RMW_IMPLEMENTATION=" "$ENV_FILE"; then
    sed -i "s|^RMW_IMPLEMENTATION=.*|RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION|" "$ENV_FILE"
else
    echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION" >> "$ENV_FILE"
fi

# Set or Update IMG_RAW_TOPIC based on TARGET_DISTRO
if grep -q -E "^IMG_RAW_TOPIC=" "$ENV_FILE"; then
    sed -i "s|^IMG_RAW_TOPIC=.*|IMG_RAW_TOPIC=$IMG_RAW_TOPIC|" "$ENV_FILE"
else
    echo "IMG_RAW_TOPIC=$IMG_RAW_TOPIC" >> "$ENV_FILE"
fi

echo "Application Docker image built successfully!"
echo "Build process completed!"
