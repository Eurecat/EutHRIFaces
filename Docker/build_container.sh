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
BASE_IMAGE="eut_ros_torch:jazzy"
CPU_ONLY="false"
REBUILD=false
NO_VCS=false
for arg in "$@"; do
    if [ "$arg" == "--clean-rebuild" ]; then
        REBUILD=true
    fi
    if [ "$arg" == "--vulcanexus" ]; then
        BASE_IMAGE="eut_ros_vulcanexus_torch:jazzy"
    fi
    if [ "$arg" == "--cpu" ]; then
        CPU_ONLY="true"
    fi
    if [ "$arg" == "--no-vcs" ]; then
        NO_VCS=true
    fi
done

# Update base image for CPU variant
if [ "$CPU_ONLY" = "true" ]; then
    if [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
        BASE_IMAGE="eut_ros_vulcanexus_torch_cpu:jazzy"
    else
        BASE_IMAGE="eut_ros_torch_cpu:jazzy"
    fi
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
    if [ "$CPU_ONLY" = "true" ]; then
        IMAGE_NAME="eut_human_face_vulcanexus_cpu:jazzy"
        echo "Building with Vulcanexus Jazzy CPU-only base image..."
    else
        IMAGE_NAME="eut_human_face_vulcanexus:jazzy"
        echo "Building with Vulcanexus Jazzy base image..."
    fi
else
    if [ "$CPU_ONLY" = "true" ]; then
        IMAGE_NAME="eut_human_face_cpu:jazzy"
        echo "Building with standard ROS2 Jazzy CPU-only base image..."
    else
        IMAGE_NAME="eut_human_face:jazzy"
        echo "Building with standard ROS2 Jazzy base image..."
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
TARGET_DISTRO="jazzy"
if grep -q -E "^TARGET_DISTRO=" "$ENV_FILE"; then
    sed -i "s/^TARGET_DISTRO=.*/TARGET_DISTRO=$TARGET_DISTRO/" "$ENV_FILE"
else
    echo "TARGET_DISTRO=$TARGET_DISTRO" >> "$ENV_FILE"
fi

echo "Application Docker image built successfully!"
echo "Build process completed!"
