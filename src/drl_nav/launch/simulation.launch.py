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

"""
Gazebo world + one robot + optional RViz.

    ros2 launch drl_nav simulation.launch.py robot:=turtlebot4 world_name:=static_world

``robot`` names a ``configs/robot/<robot>.yaml``; its ``package`` provides
``launch/spawn.launch.py`` (arguments: namespace, use_sim_time, robot_name, world, use_camera,
sensor_profile, x, y, z, yaw) and ``rviz/navigation.rviz``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, Substitution
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


class RobotPackage(Substitution):
    """ROS package of a robot, read from its configs/robot YAML."""

    def __init__(self, robot):
        super().__init__()
        self.robot = robot

    def perform(self, context):
        from drl_nav.robots.spec import RobotSpec

        return RobotSpec.load(self.robot.perform(context)).package


def generate_launch_description():
    worlds_dir = get_package_share_directory("drl_nav_worlds")

    robot = LaunchConfiguration("robot")
    robot_pkg = FindPackageShare(RobotPackage(robot))
    world_name = LaunchConfiguration("world_name")
    world = LaunchConfiguration("world")
    use_sim_time = LaunchConfiguration("use_sim_time")
    namespace = LaunchConfiguration("namespace")

    declare_args = [
        DeclareLaunchArgument(
            "robot",
            default_value="turtlebot4",
            description="Robot name (configs/robot/<robot>.yaml)",
        ),
        DeclareLaunchArgument(
            "robot_name", default_value=robot, description="Model name in Gazebo"
        ),
        DeclareLaunchArgument(
            "world_name",
            default_value="static_world",
            description="World in drl_nav_worlds/worlds (without .sdf)",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution(
                [
                    worlds_dir,
                    "worlds",
                    PythonExpression(["'", world_name, "' + '.sdf'"]),
                ]
            ),
            description="Full path to an SDF world (overrides world_name)",
        ),
        DeclareLaunchArgument(
            "namespace", default_value="", description="Top-level namespace"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="True"),
        DeclareLaunchArgument(
            "headless",
            default_value="False",
            description="No Gazebo GUI",
            choices=["True", "False", "true", "false"],
        ),
        DeclareLaunchArgument("use_rviz", default_value="True"),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=PathJoinSubstitution([robot_pkg, "rviz", "navigation.rviz"]),
        ),
        DeclareLaunchArgument(
            "use_camera",
            default_value="True",
            description="Simulate the camera (costly to render)",
            choices=["True", "False", "true", "false"],
        ),
        DeclareLaunchArgument(
            "sensor_profile",
            default_value="benchmark",
            description="Robot sensor set: 'official' or 'benchmark'",
        ),
        DeclareLaunchArgument(
            "gz_log_level", default_value="1", choices=["0", "1", "2", "3", "4"]
        ),
        DeclareLaunchArgument(
            "do_slam",
            default_value="false",
            description="Run slam_toolbox to map a new world",
            choices=["true", "false", "True", "False"],
        ),
        DeclareLaunchArgument(
            "slam_params_file",
            default_value=os.path.join(
                worlds_dir, "config", "mapper_params_online_async.yaml"
            ),
        ),
        *[
            DeclareLaunchArgument(k, default_value=v)
            for k, v in (("x", "0.0"), ("y", "0.0"), ("z", "0.01"), ("yaw", "0.0"))
        ],
    ]

    gz_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(worlds_dir, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_world": world,
            "gz_headless_mode": LaunchConfiguration("headless"),
            "gz_log_level": LaunchConfiguration("gz_log_level"),
        }.items(),
    )

    world_base = PythonExpression(["'", world, "'.split('/')[-1].split('.')[0]"])
    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([robot_pkg, "launch", "spawn.launch.py"])
        ),
        launch_arguments={
            "namespace": namespace,
            "use_sim_time": use_sim_time,
            "robot_name": LaunchConfiguration("robot_name"),
            "world": world_base,
            "use_camera": LaunchConfiguration("use_camera"),
            "sensor_profile": LaunchConfiguration("sensor_profile"),
            **{k: LaunchConfiguration(k) for k in ("x", "y", "z", "yaw")},
        }.items(),
    )

    # Clock and teleport service; stepping, pausing and queries come from gzserver's
    # simulation_interfaces.
    world_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="bridge_world",
        namespace=namespace,
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            ["/world/", world_base, "/set_pose@ros_gz_interfaces/srv/SetEntityPose"],
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch",
                "online_async_launch.py",
            )
        ),
        launch_arguments={
            "slam_params_file": LaunchConfiguration("slam_params_file")
        }.items(),
        condition=IfCondition(LaunchConfiguration("do_slam")),
    )

    rviz = Node(
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config_file")],
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    return LaunchDescription(
        [*declare_args, gz_world, world_bridge, spawn_robot, slam, rviz]
    )
