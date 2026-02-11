# Choose the base image at build time. Defaults to Vulcanexus.
ARG BASE_IMAGE="eut_ros_vulcanexus_torch:jazzy"
ARG CPU_ONLY="false"

FROM ${BASE_IMAGE}
#https://github.com/Eurecat/EutRobAIDockers

# (Optional) re-declare the args if you want to use them in later instructions
ARG BASE_IMAGE
ARG CPU_ONLY

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install CUDA dependencies only if not CPU_ONLY
RUN if [ "$CPU_ONLY" = "false" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    $(if [ "$CPU_ONLY" = "false" ]; then echo "libcudnn9-cuda-12 cuda-libraries-12-9"; fi) \
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

# Copy all requirements files
COPY requirements.txt /workspace/requirements.txt
COPY requirements_detailed.txt /workspace/requirements_detailed.txt
COPY requirements_detailed_cpu.txt /workspace/requirements_detailed_cpu.txt

# Copy requirements files and install Python dependencies in the virtual environment
# Use CPU or GPU detailed requirements based on build arg
RUN if [ "$CPU_ONLY" = "true" ]; then \
        cp requirements_detailed_cpu.txt requirements_detailed_active.txt; \
    else \
        cp requirements_detailed.txt requirements_detailed_active.txt; \
    fi

# Install packages in the ros_python_env
RUN /bin/bash -c "source /opt/ros_python_env/bin/activate && uv pip install -r requirements_detailed_active.txt"

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

# Copy coverage script for automated testing within CI/CD
COPY ci_cd_coverage.sh /ci_cd_coverage.sh
RUN chmod +x /ci_cd_coverage.sh

# Copy coverage script for automated testing within CI/CD
COPY quick_test_coverage.sh /quick_test_coverage.sh
RUN chmod +x /quick_test_coverage.sh

# Copy and set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]