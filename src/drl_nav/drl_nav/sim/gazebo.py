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

"""ROS 2 node giving the environment sensor data, commands and simulation control."""

import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from drl_nav.sensors.lidar import ScanSpec
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.client import Client
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Contacts
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import LaserScan
from simulation_interfaces.msg import Result, SimulationState
from simulation_interfaces.srv import (
    GetEntities,
    GetEntityState,
    GetSimulationState,
    SetSimulationState,
    StepSimulation,
)
from visualization_msgs.msg import Marker, MarkerArray


class SimulationError(RuntimeError):
    """Raised when the simulator does not respond or rejects a request."""


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class GazeboInterface(Node):
    """Synchronous ROS 2 client for a Gazebo simulation driven through ros_gz."""

    def __init__(
        self,
        world_name: str,
        robot_name: str,
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        cmd_vel_topic: str = "/cmd_vel",
        contact_topics: Sequence[str] = (),
        sim_interfaces_ns: str = "/gzserver",
        world_frame: str = "odom",
        base_frame: str = "base_link",
        node_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            node_name=node_name or f"{robot_name}_gz_interface",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.world_name = world_name
        self.robot_name = robot_name
        self.world_frame = world_frame
        self.base_frame = base_frame

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        self._scan: Optional[LaserScan] = None
        self._scan_spec: Optional[ScanSpec] = None
        self._scan_period: Optional[float] = None
        self._odom: Optional[Odometry] = None
        self._odom_period: Optional[float] = None
        self._last_contact_time: float = -math.inf
        self._contact_names: Tuple[str, str] = ("", "")

        self.create_subscription(
            LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, odom_topic, self._on_odom, qos_profile_sensor_data
        )
        for topic in contact_topics:
            self.create_subscription(
                Contacts, topic, self._on_contacts, qos_profile_sensor_data
            )

        self._cmd_vel_pub = self.create_publisher(TwistStamped, cmd_vel_topic, 10)
        self._marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        self._marker_array_pub = self.create_publisher(
            MarkerArray, "/visualization_marker_array", 10
        )
        self._path_pub = self.create_publisher(Path, "/path", 10)
        self._path = Path()
        self._path.header.frame_id = world_frame

        ns = sim_interfaces_ns.rstrip("/")
        self._step_client = self.create_client(StepSimulation, f"{ns}/step_simulation")
        self._set_state_client = self.create_client(
            SetSimulationState, f"{ns}/set_simulation_state"
        )
        self._get_state_client = self.create_client(
            GetSimulationState, f"{ns}/get_simulation_state"
        )
        self._get_entities_client = self.create_client(
            GetEntities, f"{ns}/get_entities"
        )
        self._get_entity_state_client = self.create_client(
            GetEntityState, f"{ns}/get_entity_state"
        )
        self._set_pose_client = self.create_client(
            SetEntityPose, f"/world/{world_name}/set_pose"
        )
        self._required_clients: List[Client] = [
            self._step_client,
            self._set_state_client,
            self._get_state_client,
            self._get_entities_client,
            self._get_entity_state_client,
            self._set_pose_client,
        ]

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self._executor.spin_once(timeout_sec=timeout_sec)

    def wait_for(
        self,
        predicate: Callable[[], bool],
        timeout_sec: float,
        what: str,
    ) -> None:
        """Spin until ``predicate()`` is true or raise :class:`SimulationError` on timeout."""
        deadline = time.monotonic() + timeout_sec
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise SimulationError(
                    f"Timed out after {timeout_sec:.1f}s waiting for {what}"
                )
            self._executor.spin_once(timeout_sec=min(0.05, remaining))

    def call(self, client: Client, request, timeout_sec: float = 5.0):
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout_sec):
                raise SimulationError(f"Service {client.srv_name} is not available")
        future = client.call_async(request)
        self._executor.spin_until_future_complete(future, timeout_sec=timeout_sec)
        if not future.done():
            future.cancel()
            raise SimulationError(
                f"Service call {client.srv_name} timed out after {timeout_sec}s"
            )
        if future.exception() is not None:
            raise SimulationError(
                f"Service call {client.srv_name} failed: {future.exception()}"
            )
        return future.result()

    def wait_for_services(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        for client in self._required_clients:
            remaining = max(0.0, deadline - time.monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                raise SimulationError(
                    f"Service {client.srv_name} did not appear within {timeout_sec:.0f}s. "
                    "Is the simulation running (ros2 launch drl_nav simulation.launch.py)?"
                )

    def sim_time(self) -> float:
        """Return the simulation time in seconds (0.0 until the first /clock message)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def now_msg(self):
        return self.get_clock().now().to_msg()

    @staticmethod
    def _check(result: Result, what: str) -> None:
        if result.result != Result.RESULT_OK:
            raise SimulationError(
                f"{what} failed with code {result.result}: {result.error_message or 'no message'}"
            )

    def set_paused(self, paused: bool) -> None:
        request = SetSimulationState.Request()
        request.state.state = (
            SimulationState.STATE_PAUSED if paused else SimulationState.STATE_PLAYING
        )
        # gzserver answers with a timeout while it is still loading heavy models; retry.
        for _ in range(5):
            response = self.call(self._set_state_client, request, timeout_sec=15.0)
            if response.result.result == Result.RESULT_OK:
                return
            self.get_logger().warning(
                f"SetSimulationState: {response.result.error_message}; retrying"
            )
            self.spin_once(1.0)
        self._check(response.result, "SetSimulationState")

    def step(self, iterations: int) -> None:
        """Request ``iterations`` physics steps. Returns as soon as Gazebo accepted the request."""
        request = StepSimulation.Request()
        request.steps = int(iterations)
        response = self.call(self._step_client, request)
        self._check(response.result, "StepSimulation")

    def advance(
        self, iterations: int, step_size: float, timeout_sec: float = 10.0
    ) -> float:
        """Step the paused simulation and block until the clock reaches the target time."""
        t_start = self.sim_time()
        t_target = t_start + iterations * step_size
        self.step(iterations)
        self.wait_for(
            lambda: self.sim_time() >= t_target - 0.5 * step_size,
            timeout_sec=timeout_sec,
            what=f"the simulation clock to advance {iterations} steps",
        )
        return self.sim_time() - t_start

    def list_entities(self, name_regex: str = "") -> List[str]:
        request = GetEntities.Request()
        request.filters.filter = name_regex
        response = self.call(self._get_entities_client, request)
        self._check(response.result, "GetEntities")
        return list(response.entities)

    def get_entity_pose(self, name: str) -> Pose:
        """Ground-truth pose of a model straight from Gazebo's entity component manager."""
        request = GetEntityState.Request()
        request.entity = name
        response = self.call(self._get_entity_state_client, request)
        self._check(response.result, f"GetEntityState({name})")
        return response.state.pose

    def teleport(self, name: str, pose: Pose) -> None:
        """Move a model with Gazebo's ``set_pose`` user command (pose only, velocity untouched)."""
        request = SetEntityPose.Request()
        request.entity.name = name
        request.pose = pose
        response = self.call(self._set_pose_client, request)
        if not response.success:
            raise SimulationError(f'Gazebo rejected set_pose for entity "{name}"')

    def publish_cmd_vel(
        self, linear_x: float, linear_y: float = 0.0, angular_z: float = 0.0
    ) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.now_msg()
        msg.header.frame_id = self.base_frame
        msg.twist = Twist()
        msg.twist.linear.x = float(linear_x)
        msg.twist.linear.y = float(linear_y)
        msg.twist.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    def _on_scan(self, msg: LaserScan) -> None:
        if self._scan is not None:
            dt = stamp_to_sec(msg.header.stamp) - stamp_to_sec(self._scan.header.stamp)
            if dt > 0:
                self._scan_period = dt
        if self._scan_spec is None:
            self._scan_spec = ScanSpec(
                angle_min=msg.angle_min,
                angle_max=msg.angle_max,
                range_min=msg.range_min,
                range_max=msg.range_max,
                num_ranges=len(msg.ranges),
            )
        self._scan = msg

    def _on_odom(self, msg: Odometry) -> None:
        if self._odom is not None:
            dt = stamp_to_sec(msg.header.stamp) - stamp_to_sec(self._odom.header.stamp)
            if dt > 0:
                self._odom_period = dt
        self._odom = msg

    def _on_contacts(self, msg: Contacts) -> None:
        for contact in msg.contacts:
            names = (contact.collision1.name, contact.collision2.name)
            if any("ground_plane" in n for n in names):
                continue
            self._last_contact_time = max(
                self._last_contact_time, stamp_to_sec(msg.header.stamp)
            )
            self._contact_names = names
            return

    @property
    def scan(self) -> Optional[LaserScan]:
        return self._scan

    @property
    def scan_spec(self) -> Optional[ScanSpec]:
        return self._scan_spec

    @property
    def scan_period(self) -> Optional[float]:
        return self._scan_period

    @property
    def odom(self) -> Optional[Odometry]:
        return self._odom

    @property
    def odom_period(self) -> Optional[float]:
        return self._odom_period

    def scan_time(self) -> float:
        return stamp_to_sec(self._scan.header.stamp) if self._scan else -math.inf

    def odom_time(self) -> float:
        return stamp_to_sec(self._odom.header.stamp) if self._odom else -math.inf

    def contact_since(self, sim_time: float) -> bool:
        return self._last_contact_time >= sim_time

    @property
    def last_contact_names(self) -> Tuple[str, str]:
        return self._contact_names

    def robot_pose(self) -> Pose:
        if self._odom is None:
            raise SimulationError("No odometry received yet")
        return self._odom.pose.pose

    def wait_for_sensors(self, timeout_sec: float) -> None:
        """Block until a scan, an odometry message and the clock have been received."""
        self.wait_for(
            lambda: (
                self._scan is not None
                and self._odom is not None
                and self.sim_time() > 0.0
            ),
            timeout_sec=timeout_sec,
            what="/scan, /odom and /clock (is the ros_gz bridge running?)",
        )

    def publish_marker(self, marker: Marker) -> None:
        self._marker_pub.publish(marker)

    def publish_marker_array(self, markers: MarkerArray) -> None:
        self._marker_array_pub.publish(markers)

    def append_path(self, pose: Pose) -> None:
        stamped = PoseStamped()
        stamped.header.frame_id = self.world_frame
        stamped.header.stamp = self.now_msg()
        stamped.pose = pose
        self._path.poses.append(stamped)
        self._path.header.stamp = stamped.header.stamp
        self._path_pub.publish(self._path)

    def clear_path(self) -> None:
        self._path.poses = []
        self._path.header.stamp = self.now_msg()
        self._path_pub.publish(self._path)

    def shutdown(self) -> None:
        self._executor.remove_node(self)
        self.destroy_node()


def diagnostics(
    world_name: str = "static_world", robot: str = "turtlebot4"
) -> Dict[str, object]:
    """Connect to a running simulation and exercise the control services."""
    from drl_nav.robots.spec import RobotSpec

    spec = RobotSpec.load(robot)
    rclpy.init()
    node = GazeboInterface(
        world_name=world_name,
        robot_name=spec.model_name,
        scan_topic=spec.topics.scan,
        odom_topic=spec.topics.odom,
        cmd_vel_topic=spec.topics.cmd_vel,
        base_frame=spec.base_frame,
    )
    try:
        node.wait_for_services(timeout_sec=10.0)
        node.set_paused(False)
        node.wait_for_sensors(timeout_sec=10.0)
        node.set_paused(True)
        before = node.sim_time()
        elapsed = node.advance(iterations=100, step_size=0.001)
        return {
            "entities": node.list_entities(),
            "scan_spec": node.scan_spec,
            "sim_time_before_step": before,
            "elapsed_after_100_steps": elapsed,
            "robot_pose": node.get_entity_pose(spec.model_name),
        }
    finally:
        node.shutdown()
        rclpy.try_shutdown()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=diagnostics.__doc__)
    parser.add_argument("--world", default="static_world")
    parser.add_argument("--robot", default="turtlebot4")
    args = parser.parse_args()
    for key, value in diagnostics(args.world, args.robot).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
