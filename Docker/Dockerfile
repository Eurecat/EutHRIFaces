# Choose the base image at build time. Defaults to Vulcanexus.
ARG BASE_IMAGE="eut_ros_vulcanexus_torch:jazzy"

FROM ${BASE_IMAGE}
#https://github.com/Eurecat/EutRobAIDockers

# (Optional) re-declare the arg if you want to use it in later instructions
ARG BASE_IMAGE

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

 # Install this for onnxgpu dependencies
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    cuda-libraries-12-9 \
    ffmpeg \
&& rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install system dependencies & development tools
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    build-essential \
    git \
&& rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install Python dependencies in the virtual environment
COPY requirements.txt /workspace/requirements.txt
RUN /bin/bash -c "source /opt/ros_python_env/bin/activate && uv pip install --no-cache -r requirements.txt"

# ------------------------------------------------------------
# Setup rosdep
# (init system-wide, update for this user)
# ------------------------------------------------------------

# Optional: fix rosdep init issue gracefully
RUN [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ] /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash;sudo rosdep init" || echo "rosdep already initialized"
RUN apt-get update && \ 
    /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash;\
      rosdep update;"
 
# Copy local ROS2 packages
COPY deps/hri_msgs /workspace/src/hri_msgs

# ------------------------------------------------------------
# Install dependencies
# (from workspace package.xmls via rosdep)
# ------------------------------------------------------------
RUN rosdep install --from-paths src --ignore-src -r -y

# ------------------------------------------------------------
# (Optional) Build workspace once during image build time
# ------------------------------------------------------------
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; \
                  cd /workspace; \
                  colcon build --event-handlers console_direct+ --symlink-install"

# Copy and set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]