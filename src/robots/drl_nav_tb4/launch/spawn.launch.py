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
Spawn the official TurtleBot4 (turtlebot4_description + Create 3 stack) into a running Gazebo.

Like turtlebot4_gz_bringup's turtlebot4_spawn.launch.py without the dock, with the world name
passed to the bridges and optional camera bridges. ``sensor_profile`` edits the generated
description in memory (drl_nav.sim.description): ``official`` keeps it as is, ``benchmark``
drops cliff/IR sensors, runs the lidar at 10 Hz / 360 rays and keeps the camera only with
``use_camera``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue

PROFILES = {
    "official": {},
    "benchmark": {
        "remove_prefixes": ("cliff_", "ir_intensity_"),
        "lidar": {"name": "rplidar", "update_rate": 10, "samples": 360},
    },
}


def robot_description(context):
    import xacro
    from drl_nav.sim.description import apply_sensor_profile

    profile = LaunchConfiguration("sensor_profile").perform(context)
    use_camera = LaunchConfiguration("use_camera").perform(context).lower() == "true"
    use_sim_time = (
        LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    )
    xacro_file = os.path.join(
        get_package_share_directory("turtlebot4_description"),
        "urdf",
        "standard",
        "turtlebot4.urdf.xacro",
    )
    namespace = LaunchConfiguration("namespace").perform(context)
    xml = xacro.process_file(
        xacro_file, mappings={"gazebo": "ignition", "namespace": namespace}
    ).toxml()
    settings = dict(PROFILES[profile])
    if profile != "official":
        settings["keep_cameras"] = use_camera
    xml = apply_sensor_profile(xml, **settings)

    remaps = [("/tf", "tf"), ("/tf_static", "tf_static")]
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            remappings=remaps,
            parameters=[
                {
                    "use_sim_time": use_sim_time,
                    "robot_description": ParameterValue(xml, value_type=str),
                }
            ],
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            output="screen",
            remappings=remaps,
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]


def generate_launch_description():
    tb4_gz = get_package_share_directory("turtlebot4_gz_bringup")
    create3_common = get_package_share_directory("irobot_create_common_bringup")
    create3_gz = get_package_share_directory("irobot_create_gz_bringup")

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    robot_name = LaunchConfiguration("robot_name")
    world = LaunchConfiguration("world")
    use_camera = LaunchConfiguration("use_camera")
    pose = {
        k: LaunchConfiguration(k, default=v)
        for k, v in (("x", "0.0"), ("y", "0.0"), ("z", "0.01"), ("yaw", "0.0"))
    }

    declare_args = [
        DeclareLaunchArgument(
            "namespace", default_value="", description="Robot namespace"
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            choices=["true", "false", "True", "False"],
        ),
        DeclareLaunchArgument(
            "robot_name", default_value="turtlebot4", description="Model name in Gazebo"
        ),
        DeclareLaunchArgument(
            "world",
            default_value="static_world",
            description="Gazebo world name (sensor topics contain it)",
        ),
        DeclareLaunchArgument(
            "use_camera",
            default_value="True",
            description="Keep and bridge the OAK-D camera",
            choices=["True", "False", "true", "false"],
        ),
        DeclareLaunchArgument(
            "sensor_profile",
            default_value="benchmark",
            choices=sorted(PROFILES),
            description="See module docstring",
        ),
    ]
    use_sim_time_lower = PythonExpression(["'", use_sim_time, "'.lower()"])

    def include(pkg_dir, name, **args):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_dir, "launch", name)),
            launch_arguments=args.items(),
        )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_robot",
        output="screen",
        arguments=[
            "-name",
            robot_name,
            "-topic",
            "robot_description",
            "-x",
            pose["x"],
            "-y",
            pose["y"],
            "-z",
            pose["z"],
            "-Y",
            pose["yaw"],
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Upstream bridges lidar, camera and HMI together; without the camera, bridge the lidar only.
    tb4_bridges = include(
        tb4_gz,
        "ros_gz_bridge.launch.py",
        model="standard",
        robot_name=robot_name,
        namespace=namespace,
        world=world,
        use_sim_time=use_sim_time_lower,
    )
    lidar_topic = [
        "/world/",
        world,
        "/model/",
        robot_name,
        "/link/rplidar_link/sensor/rplidar/scan",
    ]
    lidar_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="lidar_bridge",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[[*lidar_topic, "@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"]],
        remappings=[(lidar_topic, "scan")],
    )
    create3_bridges = include(
        create3_gz,
        "create3_ros_gz_bridge.launch.py",
        robot_name=robot_name,
        namespace=namespace,
        world=world,
        use_sim_time=use_sim_time_lower,
    )

    tb4_nodes = include(
        tb4_gz,
        "turtlebot4_nodes.launch.py",
        model="standard",
        param_file=os.path.join(tb4_gz, "config", "turtlebot4_node.yaml"),
    )
    create3_nodes = include(
        create3_common, "create3_nodes.launch.py", namespace=namespace
    )
    create3_gz_nodes = include(
        create3_gz, "create3_gz_nodes.launch.py", robot_name=robot_name
    )

    sensor_frames = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="rplidar_stf",
            arguments=[
                "0",
                "0",
                "0",
                "0",
                "0",
                "0",
                "rplidar_link",
                [robot_name, "/rplidar_link/rplidar"],
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_stf",
            arguments=[
                "0",
                "0",
                "0",
                "1.5707",
                "-1.5707",
                "0",
                "oakd_rgb_camera_optical_frame",
                [robot_name, "/oakd_rgb_camera_frame/rgbd_camera"],
            ],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ]

    group = GroupAction(
        [
            PushRosNamespace(namespace),
            OpaqueFunction(function=robot_description),
            spawn,
            create3_bridges,
            tb4_nodes,
            create3_nodes,
            create3_gz_nodes,
            *sensor_frames,
        ]
    )
    use_camera_py = PythonExpression(["'", use_camera, "'.lower() == 'true'"])
    with_camera = GroupAction(
        [PushRosNamespace(namespace), tb4_bridges], condition=IfCondition(use_camera_py)
    )
    without_camera = GroupAction(
        [PushRosNamespace(namespace), lidar_bridge],
        condition=IfCondition(PythonExpression(["not ", use_camera_py])),
    )
    return LaunchDescription([*declare_args, group, with_camera, without_camera])
