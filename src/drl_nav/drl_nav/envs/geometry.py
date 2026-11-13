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

"""Planar geometry helpers shared by the environment and its tests."""

import math
from typing import Tuple


def wrap_to_pi(angle: float) -> float:
    """Normalise an angle to ``[-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Yaw (rotation about z) of a quaternion given in (x, y, z, w) order."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) for a pure rotation about z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def relative_goal(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    goal_x: float,
    goal_y: float,
    zero_tol: float = 1e-3,
) -> Tuple[float, float]:
    """
    Distance and bearing from the robot to the goal.

    The bearing is expressed in the robot frame (0 means straight ahead, positive to the left)
    and normalised to ``[-pi, pi]``. When the robot is on top of the goal the bearing is 0.
    """
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = math.hypot(dx, dy)
    if distance < zero_tol:
        return 0.0, 0.0
    bearing = math.atan2(dy, dx) - robot_yaw
    return distance, wrap_to_pi(bearing)
