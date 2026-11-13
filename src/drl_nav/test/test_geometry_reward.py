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

from drl_nav.envs.geometry import (
    quaternion_from_yaw,
    relative_goal,
    wrap_to_pi,
    yaw_from_quaternion,
)
from drl_nav.envs.reward import compute_reward, RewardConfig
import pytest


@pytest.mark.parametrize("yaw", [0.0, 0.5, -0.5, math.pi - 0.01, -math.pi + 0.01])
def test_yaw_quaternion_round_trip(yaw):
    assert yaw_from_quaternion(*quaternion_from_yaw(yaw)) == pytest.approx(yaw)


def test_wrap_to_pi():
    # +-pi are both valid representations of a half turn.
    assert abs(wrap_to_pi(3 * math.pi)) == pytest.approx(math.pi)
    assert abs(wrap_to_pi(-3 * math.pi)) == pytest.approx(math.pi)
    assert wrap_to_pi(2 * math.pi + 0.1) == pytest.approx(0.1)
    assert wrap_to_pi(-2 * math.pi - 0.1) == pytest.approx(-0.1)
    assert wrap_to_pi(0.3) == pytest.approx(0.3)


def test_relative_goal_in_robot_frame():
    # Robot at origin facing +y; goal straight ahead.
    dist, bearing = relative_goal(0.0, 0.0, math.pi / 2, 0.0, 2.0)
    assert dist == pytest.approx(2.0)
    assert bearing == pytest.approx(0.0)
    # Goal to the robot's left (positive bearing).
    dist, bearing = relative_goal(0.0, 0.0, 0.0, 0.0, 1.0)
    assert bearing == pytest.approx(math.pi / 2)
    # Goal behind: bearing is normalised to [-pi, pi].
    _, bearing = relative_goal(1.0, 1.0, 0.0, -1.0, 1.0)
    assert abs(bearing) == pytest.approx(math.pi)


def test_relative_goal_on_top_of_goal():
    assert relative_goal(1.0, 1.0, 0.7, 1.0, 1.0) == (0.0, 0.0)


def test_reward_terminal_terms():
    cfg = RewardConfig()
    assert compute_reward(1.0, 0.0, 5.0, 0.0, True, False, cfg) == 100.0
    assert compute_reward(1.0, 0.0, 0.1, 0.0, False, True, cfg) == -100.0
    # Goal takes precedence when both flags are set.
    assert compute_reward(1.0, 0.0, 0.1, 0.0, True, True, cfg) == 100.0


def test_reward_shaping_matches_documented_formula():
    cfg = RewardConfig()
    # Far from obstacles: progress and action terms only.
    assert compute_reward(0.8, -0.2, 3.0, 0.05, False, False, cfg) == pytest.approx(
        10.0 * 0.05 + 0.5 * 0.8 - 0.5 * 0.2 - 0.001
    )
    # Moving away from the goal is penalised.
    assert compute_reward(0.0, 0.0, 3.0, -0.05, False, False, cfg) == pytest.approx(
        -0.5 - 0.001
    )
    # Close to an obstacle: negative shaping proportional to the intrusion.
    assert compute_reward(0.0, 0.0, 0.5, 0.0, False, False, cfg) == pytest.approx(
        0.5 * (0.5 - 1.0) - 0.001
    )
    # Exactly at the threshold the obstacle term is zero.
    assert compute_reward(0.0, 0.0, 1.0, 0.0, False, False, cfg) == pytest.approx(
        -0.001
    )
