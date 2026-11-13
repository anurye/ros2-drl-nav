# Multi-arch (amd64/arm64) dev image; mirrors althack/ros2:jazzy-dev's user setup.
FROM ros:jazzy-ros-base

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG WORKSPACE=/workspaces/ros2-drl-nav

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

# Optional Ubuntu mirror (e.g. http://mirrors.kernel.org/ubuntu-ports) when the default is slow.
ARG UBUNTU_MIRROR=
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
      sed -i "s|http://ports.ubuntu.com/ubuntu-ports|$UBUNTU_MIRROR|g; s|http://archive.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g; s|http://security.ubuntu.com/ubuntu|$UBUNTU_MIRROR|g" \
        /etc/apt/sources.list.d/ubuntu.sources; fi

# Non-root user `ros` (uid 1000) with sudo.
RUN if getent passwd ubuntu > /dev/null 2>&1; then userdel -r ubuntu; fi \
  && groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && apt-get update && apt-get install -y --no-install-recommends sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

# ROS 2 dev tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    cmake \
    curl \
    gdb \
    git \
    locales \
    openssh-client \
    python3-argcomplete \
    python3-pip \
    ros-dev-tools \
    ros-jazzy-ament-cmake-* \
    ros-jazzy-ament-lint-auto \
    ros-jazzy-ament-lint-common \
    ros-jazzy-ament-cppcheck \
    ros-jazzy-ament-cpplint \
    ros-jazzy-ament-flake8 \
    ros-jazzy-ament-lint-cmake \
    ros-jazzy-ament-pep257 \
    ros-jazzy-ament-uncrustify \
    ros-jazzy-ament-xmllint \
    ros-jazzy-ament-copyright \
    vim \
    wget \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*

# Gazebo Harmonic, ros_gz with simulation_interfaces, robot descriptions, Xvfb for headless
# lidar rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    mesa-utils \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-turtlebot4-description \
    ros-jazzy-turtlebot4-gz-bringup \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-ros-gz \
    ros-jazzy-rviz2 \
    ros-jazzy-simulation-interfaces \
    ros-jazzy-slam-toolbox \
    ros-jazzy-xacro \
    xvfb \
  # Browser-accessible desktop for GUIs without a host X server (macOS, remote machines).
    novnc \
    openbox \
    tigervnc-standalone-server \
    websockify \
  && rm -rf /var/lib/apt/lists/*

# Python deps in the system interpreter (rclpy lives there).
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-matplotlib \
    python3-opencv \
    python3-pytest \
    python3-scipy \
    python3-transforms3d \
    python3-yaml \
  && rm -rf /var/lib/apt/lists/* \
  # pip cannot replace Debian-installed modules; keep typing_extensions/tqdm out of apt.
  && pip install --no-cache-dir --ignore-installed typing_extensions \
  # setuptools >= 80 breaks ament_python builds (no pkg_resources / develop).
  && pip install --no-cache-dir \
    "setuptools<80" \
    "gymnasium>=1.1.1" \
    "hydra-core>=1.3" \
    "huggingface_hub" \
    "huggingface_sb3" \
    "rich>=13" \
    "stable-baselines3>=2.6.0" \
    "tensorboard>=2.19.0" \
    "tqdm>=4.67.1" \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN rosdep init || echo "rosdep already initialized"

# Shell setup for `ros`.
RUN echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc \
  && echo "if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash; fi" >> /home/$USERNAME/.bashrc \
  && echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/$USERNAME/.bashrc

ENV DEBIAN_FRONTEND=dialog
ENV AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1
# Set to 0 when a GPU is passed through.
ENV LIBGL_ALWAYS_SOFTWARE=1

# No USER here: GitHub Docker actions need root. devcontainer/compose switch to `ros`.
