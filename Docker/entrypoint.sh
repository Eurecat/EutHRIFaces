#!/bin/bash
set -e

echo "=== ENTRYPOINT START $(date) PID=$$ ==="

# Create timestamped runtime log directory for ROS2 node logs
# Use PACKAGE_NAME if set, otherwise use HOSTNAME
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
PACKAGE_DIR=${PACKAGE_NAME:-${HOSTNAME}}
BASE_RUNTIME_DIR="/workspace/log/runtime_${TIMESTAMP}"
RUNTIME_LOG_DIR="${BASE_RUNTIME_DIR}/${PACKAGE_DIR}"
mkdir -p "$RUNTIME_LOG_DIR"
echo "Created runtime log directory: $RUNTIME_LOG_DIR"

# Set ROS 2 log directory to use our timestamped folder for runtime logs
export ROS_LOG_DIR="$RUNTIME_LOG_DIR"
echo "ROS_LOG_DIR set to: $ROS_LOG_DIR"

# Create a "latest" symlink to the current runtime log directory (only from first container) (KEEP ONLY LAST 2 LOGS!!!)
if [ ! -e "/workspace/log/runtime_latest" ]; then
    ln -sf "runtime_${TIMESTAMP}" "/workspace/log/runtime_latest"
    echo "Created symlink: runtime_latest -> runtime_${TIMESTAMP}"
fi

# Source ROS 2 environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo "Sourcing ROS 2 environment..."
    source /opt/ros/jazzy/setup.bash
    echo "Sourced ${ROS_DISTRO}"
fi
if [ -f "/opt/vulcanexus/jazzy/setup.bash" ]; then
    echo "Sourcing ROS 2 environment..."
    source /opt/vulcanexus/jazzy/setup.bash
    echo "Sourced ${ROS_DISTRO}"
fi

# Source the workspace if it exists
if [ -f "/workspace/install/setup.bash" ]; then
    echo "Sourcing workspace environment..."
    source /workspace/install/setup.bash
fi

# Build face detection packages
echo "Building ros2 packages of this repo..."
cd /workspace
# rm -rf build/ install/
colcon build --event-handlers console_direct+ --symlink-install 

# Cleanup: Keep only the last 2 build and runtime logs
echo "Cleaning up old logs (keeping last 2)..."
cd /workspace/log

# Keep only last 2 build logs (excluding symlinks and COLCON_IGNORE)
BUILD_COUNT=$(ls -dt build_* 2>/dev/null | wc -l)
if [ "$BUILD_COUNT" -gt 2 ]; then
    ls -dt build_* | tail -n +3 | xargs rm -rf
    echo "Removed old build logs, kept last 2"
fi

# Keep only last 2 runtime logs (excluding symlinks)
RUNTIME_COUNT=$(ls -dt runtime_20* 2>/dev/null | wc -l)
if [ "$RUNTIME_COUNT" -gt 2 ]; then
    ls -dt runtime_20* | tail -n +3 | xargs rm -rf
    echo "Removed old runtime logs, kept last 2"
fi

cd /workspace

# Source the updated workspace after building
if [ -f "/workspace/install/setup.bash" ]; then
    echo "Sourcing updated workspace environment..."
    source /workspace/install/setup.bash
fi

echo "=== ENTRYPOINT END $(date) ==="

printenv | grep -E 'RMW|CYCLONEDDS|ROS_DOMAIN'

# Export ROS_LOG_DIR for the command that follows
# This ensures ros2 launch commands will use our timestamped directory
export ROS_LOG_DIR="$RUNTIME_LOG_DIR"

# Execute the CMD or any arguments passed to the container
exec "$@"
