#!/usr/bin/env python3

# Copyright 2025 Ahmed Yesuf Nurye.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Start Gazebo as the ros_gz ``gzserver`` node (/gzserver/* services) plus an optional GUI."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gz_gui = LaunchConfiguration("gz_gui")
    gz_headless_mode = LaunchConfiguration("gz_headless_mode")
    gz_log_level = LaunchConfiguration("gz_log_level")
    gz_world = LaunchConfiguration("gz_world")

    declare_gz_gui = DeclareLaunchArgument(
        "gz_gui",
        default_value=PathJoinSubstitution(
            [FindPackageShare("drl_nav_worlds"), "config", "teleop.config"]
        ),
        description="GUI layout configuration file.",
    )
    declare_gz_headless_mode = DeclareLaunchArgument(
        "gz_headless_mode",
        default_value="False",
        description="Do not start the Gazebo GUI client.",
        choices=["True", "False", "true", "false"],
    )
    declare_gz_log_level = DeclareLaunchArgument(
        "gz_log_level",
        default_value="1",
        description="Gazebo console verbosity (0=none ... 4=debug).",
        choices=["0", "1", "2", "3", "4"],
    )
    declare_gz_world_arg = DeclareLaunchArgument(
        "gz_world",
        default_value=PathJoinSubstitution(
            [FindPackageShare("drl_nav_worlds"), "worlds", "static_world.sdf"]
        ),
        description="Absolute path to the SDF world file.",
    )

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_server.launch.py"]
            )
        ),
        launch_arguments={
            "world_sdf_file": gz_world,
            "verbosity_level": gz_log_level,
        }.items(),
    )

    gz_gui_client = ExecuteProcess(
        cmd=["gz", "sim", "-g", "-v", gz_log_level, "--gui-config", gz_gui],
        output="screen",
        condition=UnlessCondition(gz_headless_mode),
    )

    return LaunchDescription(
        [
            declare_gz_gui,
            declare_gz_headless_mode,
            declare_gz_log_level,
            declare_gz_world_arg,
            gz_server,
            gz_gui_client,
        ]
    )
