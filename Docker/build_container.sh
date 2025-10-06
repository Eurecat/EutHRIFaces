#!/usr/bin/env bash
#
# Build script for EUT HRI Human Face Docker container
#
# Usage:
# - Default (Vulcanexus): ./build_container.sh
# - Standard ROS2: ./build_container.sh --standard-ros
# - Clean rebuild: ./build_container.sh --clean-rebuild [--standard-ros]
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
REBUILD=false
for arg in "$@"; do
    if [ "$arg" == "--clean-rebuild" ]; then
        REBUILD=true
    fi
    if [ "$arg" == "--vulcanexus" ]; then
        BASE_IMAGE="eut_ros_vulcanexus_torch:jazzy"
    fi
done

if $REBUILD; then
    echo "Rebuilding: cleaning up dependencies..."
    rm -rf $DEPS_DIR/*
fi

# Import/update dependencies repository using VCS tools (currently empty)
echo "Importing/updating dependencies repository using VCS..."
if [ -s deps.repos ]; then
    vcs import ${DEPS_DIR} < deps.repos
    vcs pull ${DEPS_DIR}
else
    echo "No external dependencies defined in deps.repos"
fi

# Display build configuration
if [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
    echo "Building with Vulcanexus Jazzy base image..."
else
    echo "Building with standard ROS2 Jazzy base image..."
fi

# Set image name based on the base image choice
if [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
    IMAGE_NAME="eut_human_face_vulcanexus:jazzy"
    echo "Building with Vulcanexus Jazzy base image..."
else
    IMAGE_NAME="eut_human_face:jazzy"
    echo "Building with standard ROS2 Jazzy base image..."
fi

echo "Base image: ${BASE_IMAGE}"
echo "Output image: ${IMAGE_NAME}"

if $REBUILD; then
    echo "Rebuilding the application Docker image..."
    docker build --no-cache . --build-arg BASE_IMAGE="${BASE_IMAGE}" -t ${IMAGE_NAME} -f Dockerfile
else
    docker build . --build-arg BASE_IMAGE="${BASE_IMAGE}" -t ${IMAGE_NAME} -f Dockerfile
fi

# Set or Update BUILT_IMAGE 
if grep -q -E "^BUILT_IMAGE=" "$ENV_FILE"; then
    sed -i "s/^BUILT_IMAGE=.*/BUILT_IMAGE=${IMAGE_NAME}/" "$ENV_FILE"
else
    echo "BUILT_IMAGE=${IMAGE_NAME}" >> "$ENV_FILE"
fi

echo "Application Docker image built successfully!"
echo "Build process completed!"
