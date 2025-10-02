#!/bin/bash
set -e

echo "=== ENTRYPOINT START $(date) PID=$$ ==="
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
colcon build --event-handlers console_direct+ 

# TODO: MAKE IT WORK SYMLINK WITHOUT NEEDING TO DELETE ALL AGAIN...
# colcon build --event-handlers console_direct+ --symlink-install

# Source the updated workspace after building
if [ -f "/workspace/install/setup.bash" ]; then
    echo "Sourcing updated workspace environment..."
    source /workspace/install/setup.bash
fi

echo "=== ENTRYPOINT END $(date) ==="

# Execute the CMD or any arguments passed to the container
exec "$@"
