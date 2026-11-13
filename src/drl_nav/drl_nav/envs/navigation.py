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

import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ament_index_python.packages import get_package_share_directory
from drl_nav.envs.geometry import (
    quaternion_from_yaw,
    relative_goal,
    yaw_from_quaternion,
)
from drl_nav.envs.reward import compute_reward, RewardConfig
from drl_nav.envs.scenarios import ScenarioGenerator
from drl_nav.robots.drives import make_drive
from drl_nav.robots.spec import resolve, RobotSpec
from drl_nav.sensors.lidar import bin_scan
from drl_nav.sim.gazebo import GazeboInterface, SimulationError
from drl_nav.sim.launcher import SimulationLauncher
from drl_nav.sim.markers import goal_marker, observation_markers
from geometry_msgs.msg import Pose
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy


class NavigationEnv(gym.Env):
    """
    Point-goal navigation with a TurtleBot4 in ROS 2 / Gazebo.

    Each ``step`` advances Gazebo by exactly ``time_delta`` seconds through the ROS 2
    ``simulation_interfaces`` services; the robot pose is Gazebo ground truth, so resets
    teleport entities without odometry drift.

    Observation (``Dict``)
        ``min_ranges`` / ``min_ranges_angle``: lidar reduced to ``num_bins`` sectors (minimum
        range and its bearing in the robot frame, [-pi, pi)); ``dist_to_goal`` [m];
        ``orient_to_goal`` [rad] in the robot frame; ``action``: previous normalised action.
    Action
        normalised velocities laid out by the robot's drive (``drl_nav.robots.drives``) and
        scaled by its ``max_speed``.
    Termination
        goal within ``goal_threshold``, or collision from the lidar (``collision_source=
        'lidar'``: reading below ``collision_threshold``), the bumper (``'contact'``) or
        either (``'both'``). Truncation comes from Gymnasium's ``TimeLimit``.
    Reset
        stops the robot for ``reset_settle_time``, samples or takes ``options['start_pos']``
        and ``options['goal_pos']`` as ``(x, y, yaw)``, teleports robot and obstacles.
        ``info['shortest_path_m']`` is the grid shortest path from start to goal.
    Reward
        see :class:`~drl_nav.envs.reward.RewardConfig`.
    Difficulty
        ``num_obstacles`` (extra ``obstacle_prefix*`` models are parked outside the arena),
        ``obstacle_clearance``, ``min_separation``/``max_separation`` (shortest-path metres).
        The registered ids ``DrlNav/TurtleBot4Easy-v0``, ``TurtleBot4-v0`` and
        ``TurtleBot4Hard-v0`` fix these (see :data:`drl_nav.TIERS`).

    Needs a running ``ros2 launch drl_nav_worlds simulation.launch.py`` or
    ``launch_simulation=True``.

    >>> env = gym.make('DrlNav/TurtleBot4-v0', launch_simulation=True)
    >>> obs, info = env.reset(seed=0)

    """

    PARKING_XY = (30.0, 30.0)

    metadata = {"render_modes": ["rviz"]}

    def __init__(
        self,
        robot: Union[str, RobotSpec, dict] = "turtlebot4",
        world_name: str = "static_world",
        map_path: Optional[Path] = None,
        yaml_path: Optional[Path] = None,
        launch_simulation: bool = False,
        launch_args: Optional[Dict[str, Any]] = None,
        robot_radius: Optional[float] = None,
        min_separation: float = 1.5,
        max_separation: Optional[float] = None,
        goal_sampling_bias: str = "uniform",
        obstacle_prefix: str = "obstacle",
        obstacle_clearance: float = 2.0,
        obstacle_radius: float = 0.25,
        num_obstacles: Optional[int] = None,
        num_bins: int = 30,
        goal_threshold: float = 0.35,
        collision_threshold: float = 0.4,
        collision_source: str = "lidar",
        time_delta: float = 0.1,
        physics_step_size: float = 0.004,
        reset_settle_time: float = 0.5,
        shuffle_on_reset: bool = True,
        reward_config: Optional[RewardConfig] = None,
        sim_interfaces_ns: str = "/gzserver",
        startup_timeout: float = 120.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if collision_source not in ("lidar", "contact", "both"):
            raise ValueError("collision_source must be 'lidar', 'contact' or 'both'")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        if time_delta <= 0 or physics_step_size <= 0:
            raise ValueError("time_delta and physics_step_size must be positive")

        self.robot_spec = resolve(robot)
        self.drive = make_drive(self.robot_spec)
        self.world_name = world_name
        self.robot_name = self.robot_spec.model_name
        maps = Path(get_package_share_directory("drl_nav_worlds")) / "maps"
        self.map_path = Path(map_path or maps / f"{world_name}.pgm")
        self.yaml_path = Path(yaml_path or maps / f"{world_name}.yaml")

        self.robot_radius = (
            self.robot_spec.footprint_radius if robot_radius is None else robot_radius
        )
        self.min_separation = min_separation
        self.max_separation = max_separation
        self.goal_sampling_bias = goal_sampling_bias
        self.obstacle_prefix = obstacle_prefix
        self.obstacle_clearance = obstacle_clearance
        self.obstacle_radius = obstacle_radius
        self.num_bins = num_bins
        self.goal_threshold = goal_threshold
        self.collision_threshold = collision_threshold
        self.collision_source = collision_source
        self.time_delta = time_delta
        self.physics_step_size = physics_step_size
        self.steps_per_action = max(1, int(round(time_delta / physics_step_size)))
        self.reset_settle_time = reset_settle_time
        self.shuffle_on_reset = shuffle_on_reset
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode

        self._launcher: Optional[SimulationLauncher] = None
        if launch_simulation:
            args = {
                "headless": True,
                "use_rviz": False,
                "use_camera": False,
                "world_name": world_name,
                "robot": self.robot_spec.name,
            }
            args.update(launch_args or {})
            self._launcher = SimulationLauncher(launch_args=args).start()

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.sim = GazeboInterface(
            world_name=world_name,
            robot_name=self.robot_name,
            scan_topic=self.robot_spec.topics.scan,
            odom_topic=self.robot_spec.topics.odom,
            cmd_vel_topic=self.robot_spec.topics.cmd_vel,
            contact_topics=self.robot_spec.topics.contacts
            if collision_source != "lidar"
            else (),
            base_frame=self.robot_spec.base_frame,
            sim_interfaces_ns=sim_interfaces_ns,
        )
        try:
            self._connect(startup_timeout)
        except Exception:
            self.close()
            raise

        spec = self.sim.scan_spec
        self.num_bins = min(self.num_bins, spec.num_ranges)
        self.action_space = self.drive.action_space()
        self.observation_space = spaces.Dict(
            {
                "min_ranges": spaces.Box(
                    low=spec.range_min,
                    high=spec.range_max,
                    shape=(self.num_bins,),
                    dtype=np.float32,
                ),
                "min_ranges_angle": spaces.Box(
                    low=-math.pi,
                    high=math.pi,
                    shape=(self.num_bins,),
                    dtype=np.float32,
                ),
                "dist_to_goal": spaces.Box(
                    low=0.0, high=100.0, shape=(1,), dtype=np.float32
                ),
                "orient_to_goal": spaces.Box(
                    low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32
                ),
                "action": spaces.Box(
                    low=self.action_space.low,
                    high=self.action_space.high,
                    dtype=np.float32,
                ),
            }
        )

        self.nav_scenario = ScenarioGenerator(
            map_path=self.map_path,
            yaml_path=self.yaml_path,
            # The lidar criterion ends an episode at collision_threshold, so plan with it.
            robot_radius=max(self.robot_radius, self.collision_threshold),
            min_separation=self.min_separation,
            max_separation=self.max_separation,
            obstacle_radius=self.obstacle_radius,
            obstacle_clearance=self.obstacle_clearance,
            rng=self.np_random,
        )
        # GetEntities matches the regex against the whole name.
        models: List[str] = self.sim.list_entities(
            f"{re.escape(self.obstacle_prefix)}.*"
        )
        if num_obstacles is None:
            num_obstacles = len(models)
        if num_obstacles > len(models):
            raise ValueError(
                f"num_obstacles={num_obstacles}, but the world has only "
                f'{len(models)} "{self.obstacle_prefix}*" models'
            )
        if self.shuffle_on_reset and not models:
            self.sim.get_logger().warning(
                f'No Gazebo models starting with "{self.obstacle_prefix}" found; '
                "obstacle shuffling is disabled."
            )
        self._obstacles = models[:num_obstacles]
        for i, name in enumerate(models[num_obstacles:]):
            x, y = self.PARKING_XY
            self.sim.teleport(name, _pose_from_xy_yaw(x + i, y, 0.0))

        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._goal_xy: Tuple[float, float] = (0.0, 0.0)
        self._goal_pose = Pose()
        self._step_start_time = 0.0
        self._episode_steps = 0
        self._prev_dist = 0.0
        self._shortest_path = float("inf")

    def _connect(self, timeout: float) -> None:
        self.sim.wait_for_services(timeout_sec=timeout)
        self.sim.set_paused(False)
        self.sim.wait_for_sensors(timeout_sec=timeout)
        # A second scan gives the scan period used for synchronisation.
        first_scan_time = self.sim.scan_time()
        self.sim.wait_for(
            lambda: self.sim.scan_time() > first_scan_time,
            timeout_sec=10.0,
            what="a second scan",
        )
        # Some robot stacks (ros2_control spawners) need running sim time to become ready.
        settle = self.robot_spec.startup_settle_time
        if settle > 0:
            t0 = self.sim.sim_time()
            self.sim.wait_for(
                lambda: self.sim.sim_time() >= t0 + settle,
                timeout_sec=10 * settle + 10,
                what="the robot stack to settle",
            )
        self.sim.set_paused(True)
        if self.sim.scan_period and self.time_delta < self.sim.scan_period - 1e-6:
            self.sim.get_logger().warning(
                f"time_delta ({self.time_delta}s) is shorter than the lidar period "
                f"({self.sim.scan_period:.3f}s): consecutive observations may share a scan."
            )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.nav_scenario.reseed(self.np_random)
        options = options or {}

        # Stop the robot before teleporting so no momentum carries over.
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.sim.publish_cmd_vel(0.0, 0.0, 0.0)
        if self.reset_settle_time > 0:
            self._advance(self.reset_settle_time)

        start_pos = options.get("start_pos")
        goal_pos = options.get("goal_pos")
        if start_pos is None or goal_pos is None:
            start_xy, goal_xy = self.nav_scenario.generate_start_goal(
                max_attempts=100, goal_sampling_bias=self.goal_sampling_bias
            )
            start_pos = start_pos or (
                *start_xy,
                self.np_random.uniform(-math.pi, math.pi),
            )
            goal_pos = goal_pos or (*goal_xy, self.np_random.uniform(-math.pi, math.pi))

        start_pose = _pose_from_xy_yaw(*start_pos)
        self._goal_pose = _pose_from_xy_yaw(*goal_pos)
        self._goal_xy = (float(goal_pos[0]), float(goal_pos[1]))

        self.sim.teleport(self.robot_name, start_pose)
        obstacles: List[Tuple[float, float]] = []
        if self.shuffle_on_reset and self._obstacles:
            obstacles = self._shuffle_obstacles(start_pos[:2], goal_pos[:2])
        self._shortest_path = self.nav_scenario.shortest_path_length(
            start_pos[:2], goal_pos[:2], obstacles
        )

        self._advance(self.time_delta)
        self._verify_teleport(start_pos)

        self.sim.clear_path()
        self.sim.publish_marker(
            goal_marker(self._goal_pose, self.sim.world_frame, self.sim.now_msg())
        )

        self._episode_steps = 0
        observation = self._get_obs()
        self._prev_dist = float(observation["dist_to_goal"][0])
        info = self._get_info(observation, goal_reached=False, collision=False)
        if self.render_mode == "rviz":
            self.render()
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self._last_action = action.copy()

        self.sim.publish_cmd_vel(*self.drive.twist(action))
        self._advance(self.time_delta)
        self._episode_steps += 1

        observation = self._get_obs()
        dist_to_goal = float(observation["dist_to_goal"][0])
        min_range = float(observation["min_ranges"].min())

        goal_reached = dist_to_goal < self.goal_threshold
        collision = self._collision(min_range)
        terminated = bool(goal_reached or collision)
        speed, turn = self.drive.effort(action)
        reward = compute_reward(
            linear_action=speed,
            angular_action=turn,
            min_range=min_range,
            progress=self._prev_dist - dist_to_goal,
            goal_reached=goal_reached,
            collision=collision,
            config=self.reward_config,
        )
        self._prev_dist = dist_to_goal
        info = self._get_info(observation, goal_reached, collision)

        self.sim.append_path(self.sim.robot_pose())
        if self.render_mode == "rviz":
            self.render()
        return observation, reward, terminated, False, info

    def render(self) -> None:
        if self.render_mode != "rviz":
            return
        observation = self._get_obs()
        self.sim.publish_marker_array(
            observation_markers(
                observation=observation,
                robot_xy_yaw=self._robot_xy_yaw(),
                goal_xy=self._goal_xy,
                world_frame=self.sim.world_frame,
                base_frame=self.sim.base_frame,
                stamp=self.sim.now_msg(),
                robot_radius=self.robot_radius,
            )
        )

    def close(self) -> None:
        sim = getattr(self, "sim", None)
        if sim is not None:
            try:
                if rclpy.ok():
                    sim.publish_cmd_vel(0.0, 0.0, 0.0)
                    sim.spin_once(0.05)
            except Exception:  # noqa: BLE001 - best effort during shutdown
                pass
            sim.shutdown()
            self.sim = None
        if getattr(self, "_owns_rclpy", False):
            rclpy.try_shutdown()
            self._owns_rclpy = False
        if self._launcher is not None:
            self._launcher.stop()
            self._launcher = None

    def _advance(self, duration: float) -> None:
        """Advance sim time by ``duration`` and wait until sensors have caught up."""
        iterations = max(1, int(round(duration / self.physics_step_size)))
        self._step_start_time = self.sim.sim_time()
        elapsed = self.sim.advance(iterations, self.physics_step_size)
        expected = iterations * self.physics_step_size
        if abs(elapsed - expected) > 0.5 * self.physics_step_size + 1e-6:
            self.sim.get_logger().warning(
                f"Simulation advanced {elapsed:.4f}s instead of {expected:.4f}s. Check that "
                f"physics_step_size matches <max_step_size> in the world file."
            )
        self._sync_sensors(self._step_start_time + elapsed)

    def _sync_sensors(self, sim_time: float) -> None:
        """Wait for the newest scan/odom produced before ``sim_time`` to be delivered."""
        scan_period = self.sim.scan_period or 0.1
        odom_period = self.sim.odom_period or 0.02
        tol = 1e-3

        def ready() -> bool:
            return (
                self.sim.scan_time() >= sim_time - scan_period - tol
                and self.sim.odom_time() >= sim_time - odom_period - tol
            )

        try:
            self.sim.wait_for(
                ready, timeout_sec=2.0, what="sensor data for the current step"
            )
        except SimulationError as exc:
            self.sim.get_logger().warning(f"{exc}; using the latest available data.")

    def _shuffle_obstacles(self, start_xy, goal_xy) -> List[Tuple[float, float]]:
        positions = self.nav_scenario.generate_obstacles(
            num_obstacles=len(self._obstacles), start_pos=start_xy, goal_pos=goal_xy
        )
        for name, (x, y) in zip(self._obstacles, positions):
            self.sim.teleport(name, _pose_from_xy_yaw(x, y, 0.0))
        return positions

    def _verify_teleport(self, start_pos, tolerance: float = 0.05) -> None:
        x, y, _ = self._robot_xy_yaw()
        error = math.hypot(x - start_pos[0], y - start_pos[1])
        if error > tolerance:
            self.sim.get_logger().warning(
                f"Robot is {error:.3f}m away from the requested start pose after reset."
            )

    def _robot_xy_yaw(self) -> Tuple[float, float, float]:
        pose = self.sim.robot_pose()
        q = pose.orientation
        return pose.position.x, pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        scan = self.sim.scan
        min_ranges, min_ranges_angle = bin_scan(
            scan.ranges,
            self.sim.scan_spec,
            self.num_bins,
            yaw_offset=self.robot_spec.lidar_yaw,
        )
        x, y, yaw = self._robot_xy_yaw()
        dist_to_goal, orient_to_goal = relative_goal(x, y, yaw, *self._goal_xy)
        return {
            "min_ranges": min_ranges,
            "min_ranges_angle": min_ranges_angle,
            "dist_to_goal": np.array([dist_to_goal], dtype=np.float32),
            "orient_to_goal": np.array([orient_to_goal], dtype=np.float32),
            "action": self._last_action.astype(np.float32),
        }

    def _collision(self, min_range: float) -> bool:
        lidar_hit = min_range < self.collision_threshold
        contact_hit = self.sim.contact_since(self._step_start_time)
        if self.collision_source == "lidar":
            return lidar_hit
        if self.collision_source == "contact":
            return contact_hit
        return lidar_hit or contact_hit

    def _get_info(
        self, observation, goal_reached: bool, collision: bool
    ) -> Dict[str, Any]:
        x, y, yaw = self._robot_xy_yaw()
        return {
            "distance_to_goal": float(observation["dist_to_goal"][0]),
            "is_success": bool(goal_reached),
            "collision": bool(collision),
            "sim_time": self.sim.sim_time(),
            "episode_steps": self._episode_steps,
            "robot_pose": (x, y, yaw),
            "goal": self._goal_xy,
            "shortest_path_m": self._shortest_path,
        }


def _pose_from_xy_yaw(x: float, y: float, yaw: float, z: float = 0.01) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    qx, qy, qz, qw = quaternion_from_yaw(float(yaw))
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def main() -> None:
    """Drive a scripted policy for a few episodes, publishing RViz markers (sanity check)."""
    import argparse

    import drl_nav  # noqa: F401

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--launch", action="store_true", help="start the simulation headless"
    )
    parser.add_argument(
        "--env", default="DrlNav/TurtleBot4-v0", help="registered env id"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--policy", choices=["straight", "random"], default="straight")
    parser.add_argument(
        "--collision-source", default="lidar", choices=["lidar", "contact", "both"]
    )
    args = parser.parse_args()

    env = gym.make(
        args.env,
        launch_simulation=args.launch,
        collision_source=args.collision_source,
        render_mode="rviz",
    )
    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            x, y, yaw = info["robot_pose"]
            print(
                f"episode {episode + 1}: start=({x:.2f}, {y:.2f}, {yaw:.2f}) "
                f"goal={tuple(round(v, 2) for v in info['goal'])} "
                f"path={info['shortest_path_m']:.2f}m"
            )
            while True:
                action = (
                    env.action_space.sample() if args.policy == "random" else [0.5, 0.0]
                )
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(
                        f"  done after {info['episode_steps']} steps: "
                        f"success={info['is_success']} collision={info['collision']} "
                        f"dist={info['distance_to_goal']:.2f}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
