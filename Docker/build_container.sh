#!/usr/bin/env bash
#
# Build script for EUT HRI Human Face Docker container
#
# Usage:
# - Default (Vulcanexus): ./build_container.sh
# - Standard ROS2: ./build_container.sh --standard-ros
# - Humble Vulcanexus: ./build_container.sh --humble --vulcanexus
# - CPU-only version: ./build_container.sh --cpu
# - CPU-only Vulcanexus: ./build_container.sh --cpu --vulcanexus
# - Jetson Thor / ARM64:  ./build_container.sh --arm
# - Clean rebuild: ./build_container.sh --clean-rebuild [--standard-ros] [--cpu] [--arm]
#
# --arm uses Dockerfile.arm and base image eut_ros_torch:jazzy (ARM).
# Build context is the EutHRIFaces repo root so COPY of ROS packages works.
# Mutually exclusive with --vulcanexus / --humble / --cpu (Jazzy + GPU only).
#

export DOCKER_BUILDKIT=1

set -e

show_help() {
    cat <<'EOF'
Usage: ./build_container.sh [OPTIONS]

Options:
  --clean-rebuild    Force no-cache Docker build and clean deps cache
  --vulcanexus       Use Vulcanexus ROS 2 base images
  --humble           Build Humble variant (x86 only unless script-specific)
  --cpu              Build CPU-only variant (no NVIDIA runtime)
  --arm              Build ARM64 / Jetson variant (Jazzy + GPU)
  --no-vcs           Skip vcs import/pull for Docker/deps
  --push-after-build Force Docker Hub push after build (overrides .env)
  --no-push-after-build Disable Docker Hub push for this run (overrides .env)
  --help, -h         Show this help and exit

Docker Hub integration (from local Docker/.env):
  DOCKERHUB_ORG or DOCKERHUB_NAMESPACE
  DOCKERHUB_USERNAME
  DOCKERHUB_TOKEN
  DOCKERHUB_REPO_PREFIX
  DOCKERHUB_PUSH_AFTER_BUILD=true  # enable push after successful build

Examples:
  ./build_container.sh
  ./build_container.sh --arm
  ./build_container.sh --cpu --clean-rebuild
EOF
    exit 0
}

# Handle help before any VCS or filesystem side effects
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        show_help
    fi
done


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
USE_ARM=false
PUSH_AFTER_BUILD_OVERRIDE=""
PLATFORM_ARCH="amd"
for arg in "$@"; do
    if [ "$arg" == "--clean-rebuild" ]; then
        REBUILD=true
    fi
    if [ "$arg" == "--vulcanexus" ]; then
        USE_VULCANEXUS=true
    fi
    if [ "$arg" == "--cpu" ]; then
        CPU_ONLY="true"
    fi
    if [ "$arg" == "--humble" ]; then
        TARGET_DISTRO="humble"
        USE_HUMBLE=true
    fi
    if [ "$arg" == "--no-vcs" ]; then
        NO_VCS=true
    fi
    if [ "$arg" == "--arm" ]; then
        USE_ARM=true
    fi
    if [ "$arg" == "--push-after-build" ]; then
        PUSH_AFTER_BUILD_OVERRIDE="true"
    fi
    if [ "$arg" == "--no-push-after-build" ]; then
        PUSH_AFTER_BUILD_OVERRIDE="false"
    fi
done

# --arm validation: mutually exclusive with --vulcanexus / --humble / --cpu
if $USE_ARM; then
    if $USE_VULCANEXUS; then
        echo "Error: --arm is not supported with --vulcanexus (ARM base is standard ROS 2 Jazzy)."
        exit 1
    fi
    if $USE_HUMBLE; then
        echo "Error: --arm is not supported with --humble (ARM base is forced to Jazzy)."
        exit 1
    fi
    if [ "$CPU_ONLY" = "true" ]; then
        echo "Error: --arm requires the Jetson GPU-enabled base image; --cpu is incompatible."
        exit 1
    fi
    TARGET_DISTRO="jazzy"
    PLATFORM_ARCH="arm"
fi

# Resolve base image from selected flags
if $USE_ARM; then
    BASE_IMAGE="eut_ros_torch:${TARGET_DISTRO}"
elif $USE_VULCANEXUS; then
    BASE_IMAGE="eut_ros_vulcanexus_torch:${TARGET_DISTRO}"
else
    BASE_IMAGE="eut_ros_torch:${TARGET_DISTRO}"
fi

# Update base image for CPU variant
if [ "$CPU_ONLY" = "true" ]; then
    if [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
        BASE_IMAGE="eut_ros_vulcanexus_torch_cpu:${TARGET_DISTRO}"
    else
        BASE_IMAGE="eut_ros_torch_cpu:${TARGET_DISTRO}"
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

# Display build configuration
if $USE_ARM; then
    echo "Building Jetson Thor / ARM64 (Jazzy + PyTorch ARM) image..."
elif [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
    echo "Building with Vulcanexus ${TARGET_DISTRO} base image..."
else
    echo "Building with standard ROS2 ${TARGET_DISTRO} base image..."
fi

# Set image name based on the base image choice and CPU/ARM flags
if $USE_ARM; then
    IMAGE_NAME="eut_human_face_arm:${TARGET_DISTRO}"
elif [[ "${BASE_IMAGE}" == *"vulcanexus"* ]]; then
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
echo "Platform: ${PLATFORM_ARCH}"
echo "CPU Only: ${CPU_ONLY}"
echo "ARM build: ${USE_ARM}"
echo "Output image: ${IMAGE_NAME}"

# Build:
# - --arm uses Dockerfile.arm with build context = repo root (..) so it can
#   COPY face_detection / face_recognition / gaze_estimation / visual_speech_activity
#   and Docker/deps/hri_msgs.
# - x86_64 path keeps existing behaviour: build context = Docker dir.
if $USE_ARM; then
    DOCKERFILE="Dockerfile.arm"
    BUILD_CONTEXT=".."
    BUILD_PLATFORM="linux/arm64"
    BUILD_ARGS=(
        --platform "${BUILD_PLATFORM}"
        --build-arg BASE_IMAGE="${BASE_IMAGE}"
        --build-arg PLATFORM_ARCH="arm"
        -t "${IMAGE_NAME}"
        -f "${DOCKERFILE}"
        "${BUILD_CONTEXT}"
    )
    if $REBUILD; then
        echo "Rebuilding ARM image (no cache)..."
        docker build --no-cache "${BUILD_ARGS[@]}"
    else
        docker build "${BUILD_ARGS[@]}"
    fi
else
    # Set Docker platform based on architecture
    DOCKER_PLATFORM="linux/amd64"
    if $REBUILD; then
        echo "Rebuilding the application Docker image..."
        docker build --platform ${DOCKER_PLATFORM} --no-cache . --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg CPU_ONLY="${CPU_ONLY}" --build-arg PLATFORM_ARCH="${PLATFORM_ARCH}" -t ${IMAGE_NAME} -f Dockerfile
    else
        docker build --platform ${DOCKER_PLATFORM} . --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg CPU_ONLY="${CPU_ONLY}" --build-arg PLATFORM_ARCH="${PLATFORM_ARCH}" -t ${IMAGE_NAME} -f Dockerfile
    fi
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

# Optional per-repo Docker Hub push driven by Docker/.env
PUSH_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dockerhub_push.sh"
if [ -x "$PUSH_HELPER" ]; then
    if [ -n "$PUSH_AFTER_BUILD_OVERRIDE" ]; then
        DOCKERHUB_PUSH_AFTER_BUILD="$PUSH_AFTER_BUILD_OVERRIDE" "$PUSH_HELPER" --env-file "$ENV_FILE" --image "$IMAGE_NAME"
    else
        "$PUSH_HELPER" --env-file "$ENV_FILE" --image "$IMAGE_NAME"
    fi
fi
