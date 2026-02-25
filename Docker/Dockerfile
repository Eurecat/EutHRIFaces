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

# NOTE: We use pip-only CUDA stack (nvidia-cudnn-cu12, etc.) from requirements_detailed.txt
# No apt CUDA/cuDNN packages needed - avoiding mixing apt and pip CUDA stacks

RUN apt-get update && apt-get install -y --no-install-recommends \
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
COPY requirements_detailed_humble.txt /workspace/requirements_detailed_humble.txt
COPY requirements_detailed.txt /workspace/requirements_detailed.txt
COPY requirements_detailed_cpu.txt /workspace/requirements_detailed_cpu.txt

# Copy requirements files and install Python dependencies in the virtual environment
# Use CPU or GPU detailed requirements based on build arg
# For Humble+CPU use requirements_detailed_humble_cpu.txt, for Humble use requirements_detailed_humble.txt, for Jazzy+CPU use requirements_detailed_cpu.txt, else requirements_detailed.txt
RUN if [ "${ROS_DISTRO}" = "humble" ]; then \
        if [ "$CPU_ONLY" = "true" ] && [ -f requirements_detailed_humble_cpu.txt ]; then \
            cp requirements_detailed_humble_cpu.txt requirements_detailed_active.txt; \
        else \
            cp requirements_detailed_humble.txt requirements_detailed_active.txt; \
        fi; \
    else \
        if [ "$CPU_ONLY" = "true" ]; then \
            cp requirements_detailed_cpu.txt requirements_detailed_active.txt; \
        else \
            cp requirements_detailed.txt requirements_detailed_active.txt; \
        fi; \
    fi

# Install packages in the ros_python_env
RUN /bin/bash -c "source /opt/ros_python_env/bin/activate && uv pip install -r requirements_detailed_active.txt"

# ------------------------------------------------------------
# Configure LD_LIBRARY_PATH for pip-installed CUDA libraries
# This ensures onnxruntime-gpu can find libcudnn_adv.so.9 and other cuDNN libs
# Python 3.10 for Humble, Python 3.12 for Jazzy
# We add it to:
# 1. /etc/profile.d/ - sourced by entrypoint and login shells
# 2. /etc/bash.bashrc - sourced by interactive bash shells (docker exec)
# 3. /etc/environment - read by PAM sessions
# ------------------------------------------------------------
# 
RUN PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    echo "Python version detected: ${PYTHON_VER}" && \
    CUDA_LIB_PATH="/opt/ros_python_env/lib/python${PYTHON_VER}/site-packages/nvidia/cudnn/lib:\
/opt/ros_python_env/lib/python${PYTHON_VER}/site-packages/nvidia/cublas/lib:\
/opt/ros_python_env/lib/python${PYTHON_VER}/site-packages/nvidia/cuda_runtime/lib:\
/opt/ros_python_env/lib/python${PYTHON_VER}/site-packages/nvidia/curand/lib:\
/opt/ros_python_env/lib/python${PYTHON_VER}/site-packages/nvidia/cufft/lib" && \
    echo "export LD_LIBRARY_PATH=${CUDA_LIB_PATH}:\${LD_LIBRARY_PATH}" >> /etc/profile.d/cuda_lib_path.sh && \
    chmod +x /etc/profile.d/cuda_lib_path.sh && \
    echo "export LD_LIBRARY_PATH=${CUDA_LIB_PATH}:\${LD_LIBRARY_PATH}" >> /etc/bash.bashrc && \
    echo "LD_LIBRARY_PATH=\"${CUDA_LIB_PATH}:\${LD_LIBRARY_PATH}\"" >> /etc/environment

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