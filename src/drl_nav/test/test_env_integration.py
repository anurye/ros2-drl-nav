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
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DRL_NAV_INTEGRATION") != "1",
    reason="set DRL_NAV_INTEGRATION=1 to run against Gazebo",
)


@pytest.fixture(scope="module")
def env():
    import gymnasium as gym
    import drl_nav  # noqa: F401

    env = gym.make(
        "DrlNav/TurtleBot4-v0",
        launch_simulation=True,
        launch_args={"gz_log_level": 1},
        collision_source="both",
        time_delta=0.1,
    )
    yield env
    env.close()


def test_reset_teleports_robot_to_requested_pose(env):
    start = (1.0, -1.5, 0.5)
    goal = (3.0, 1.0, 0.0)
    obs, info = env.reset(seed=1, options={"start_pos": start, "goal_pos": goal})
    x, y, yaw = info["robot_pose"]
    assert math.hypot(x - start[0], y - start[1]) < 0.05
    assert abs(math.atan2(math.sin(yaw - start[2]), math.cos(yaw - start[2]))) < 0.05
    assert info["distance_to_goal"] == pytest.approx(math.hypot(2.0, 2.5), abs=0.1)
    assert env.observation_space.contains(obs)


def test_step_advances_sim_time_deterministically(env):
    _, info = env.reset(seed=2)
    t0 = info["sim_time"]
    for i in range(1, 6):
        _, _, _, _, info = env.step([0.0, 0.0])
        assert info["sim_time"] - t0 == pytest.approx(0.1 * i, abs=0.002)


def test_robot_moves_forward_when_commanded(env):
    _, info = env.reset(
        seed=3, options={"start_pos": (0.0, 0.0, 0.0), "goal_pos": (2.0, 0.0, 0.0)}
    )
    x0 = info["robot_pose"][0]
    for _ in range(15):
        _, _, terminated, _, info = env.step([1.0, 0.0])
        if terminated:
            break
    assert info["robot_pose"][0] > x0 + 0.2


def test_reset_samples_reachable_scenario(env):
    base = env.unwrapped
    _, info = env.reset(seed=6)
    assert len(base._obstacles) == 12
    assert base.min_separation <= info["shortest_path_m"] < math.inf
    assert len(base.sim.list_entities("obstacle.*")) == 20


def test_lidar_bearings_are_in_the_robot_frame(env):
    # Facing the east wall (x = 5.925) from 1.4 m; the north wall is 2.9 m away. The nearest
    # sector must be straight ahead whatever the lidar's mounting yaw.
    obs, _ = env.reset(
        seed=7, options={"start_pos": (4.5, 3.0, 0.0), "goal_pos": (-3.0, -3.0, 0.0)}
    )
    nearest = int(np.argmin(obs["min_ranges"]))
    assert obs["min_ranges"][nearest] == pytest.approx(1.4, abs=0.15)
    assert abs(obs["min_ranges_angle"][nearest]) < 0.25


def test_seeding_is_reproducible(env):
    _, a = env.reset(seed=11)
    _, b = env.reset(seed=11)
    assert a["goal"] == pytest.approx(b["goal"])
    assert a["robot_pose"][:2] == pytest.approx(b["robot_pose"][:2], abs=0.05)


def test_episode_terminates_on_collision_with_wall(env):
    # Free cell facing the north wall (y = 6); the lidar threshold fires before contact.
    env.reset(
        seed=4,
        options={"start_pos": (3.0, 3.0, math.pi / 2), "goal_pos": (-3.0, -3.0, 0.0)},
    )
    terminated = False
    for _ in range(100):
        _, reward, terminated, _, info = env.step([1.0, 0.0])
        if terminated:
            break
    assert terminated
    assert info["collision"]
    assert reward == -100.0
    assert info["robot_pose"][1] < 5.7


def test_bumper_contact_terminates_episode(env):
    # Lidar criterion disabled: only the bumper ends the episode.
    base = env.unwrapped
    previous = base.collision_source
    base.collision_source = "contact"
    try:
        env.reset(
            seed=5,
            options={
                "start_pos": (3.0, 4.5, math.pi / 2),
                "goal_pos": (-3.0, -3.0, 0.0),
            },
        )
        terminated = False
        for _ in range(80):
            _, reward, terminated, _, info = env.step([1.0, 0.0])
            if terminated:
                break
    finally:
        base.collision_source = previous
    assert terminated
    assert info["collision"]
    assert reward == -100.0
    # Touching the wall face at y = 5.925.
    assert info["robot_pose"][1] > 5.7
    assert "Wall" in "".join(base.sim.last_contact_names)
