#!/bin/bash
set -e

echo "=== ENTRYPOINT START $(date) PID=$$ ==="

# Source CUDA library path if it exists (for pip-installed CUDA libraries)
if [ -f /etc/profile.d/cuda_lib_path.sh ]; then
    echo "Sourcing CUDA library path..."
    source /etc/profile.d/cuda_lib_path.sh
fi

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
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    echo "Sourcing ROS 2 environment..."
    source /opt/ros/${ROS_DISTRO}/setup.bash
    echo "Sourced ${ROS_DISTRO}"
fi
if [ -f "/opt/vulcanexus/${ROS_DISTRO}/setup.bash" ]; then
    echo "Sourcing ROS 2 environment..."
    source /opt/vulcanexus/${ROS_DISTRO}/setup.bash
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
# --base-paths restricts package discovery to src/ only, preventing colcon from
# scanning build/ or install/ and picking up stale CMake packages (e.g. LibreYOLO internals).
colcon build --base-paths /workspace/src --event-handlers console_direct+ --symlink-install

# Cleanup: Keep only the last N build/runtime logs
# BUILD_LOGS_TO_KEEP:   number of build log sessions to retain.   -1 = never delete.
# RUNTIME_LOGS_TO_KEEP: number of runtime log sessions to retain. -1 = never delete.
BUILD_LOGS_TO_KEEP=${BUILD_LOGS_TO_KEEP:-2}
RUNTIME_LOGS_TO_KEEP=${RUNTIME_LOGS_TO_KEEP:-2}
echo "Cleaning up old logs (build: keep last ${BUILD_LOGS_TO_KEEP}, runtime: keep last ${RUNTIME_LOGS_TO_KEEP}; -1 = keep all)..."
cd /workspace/log

# Keep only last N build logs
if [ "$BUILD_LOGS_TO_KEEP" -ne -1 ]; then
    BUILD_COUNT=$(ls -dt build_* 2>/dev/null | wc -l)
    if [ "$BUILD_COUNT" -gt "$BUILD_LOGS_TO_KEEP" ]; then
        ls -dt build_* | tail -n +"$((BUILD_LOGS_TO_KEEP + 1))" | xargs rm -rf
        echo "Removed old build logs, kept last ${BUILD_LOGS_TO_KEEP}"
    fi
else
    echo "Build log retention disabled (BUILD_LOGS_TO_KEEP=-1), keeping all build logs."
fi

# Keep only last N runtime logs
if [ "$RUNTIME_LOGS_TO_KEEP" -ne -1 ]; then
    RUNTIME_COUNT=$(ls -dt runtime_20* 2>/dev/null | wc -l)
    if [ "$RUNTIME_COUNT" -gt "$RUNTIME_LOGS_TO_KEEP" ]; then
        ls -dt runtime_20* | tail -n +"$((RUNTIME_LOGS_TO_KEEP + 1))" | xargs rm -rf
        echo "Removed old runtime logs, kept last ${RUNTIME_LOGS_TO_KEEP}"
    fi
else
    echo "Runtime log retention disabled (RUNTIME_LOGS_TO_KEEP=-1), keeping all runtime logs."
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
