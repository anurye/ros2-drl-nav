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

"""Drive kinematics: normalised action -> Twist. Add a class here for a new drive type."""

from typing import Dict, List, Sequence, Tuple, Type

from drl_nav.robots.spec import RobotSpec
from gymnasium import spaces
import numpy as np

Twist = Tuple[float, float, float]  # linear.x, linear.y, angular.z


class Drive:
    """Base class; subclasses define the action layout."""

    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.v_min = -1.0 if spec.allow_reverse else 0.0

    def action_space(self) -> spaces.Box:
        raise NotImplementedError

    def twist(self, action: np.ndarray) -> Twist:
        raise NotImplementedError

    def effort(self, action: np.ndarray) -> Tuple[float, float]:
        """Return (normalised speed, normalised turn rate) for the reward terms."""
        raise NotImplementedError

    def discrete_actions(self) -> List[Sequence[float]]:
        """Return the default action set for discrete-action algorithms (DQN)."""
        raise NotImplementedError


class DiffDrive(Drive):
    """Action ``[v, w]``: forward and angular velocity."""

    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array([self.v_min, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def twist(self, action: np.ndarray) -> Twist:
        s = self.spec.max_speed
        return float(action[0]) * s["linear"], 0.0, float(action[1]) * s["angular"]

    def effort(self, action: np.ndarray) -> Tuple[float, float]:
        return abs(float(action[0])), float(action[1])

    def discrete_actions(self) -> List[Sequence[float]]:
        return [[1, 0], [1, 0.5], [1, -0.5], [0, 1], [0, -1]]


class OmniDrive(Drive):
    """Action ``[vx, vy, w]``: forward, lateral and angular velocity (mecanum/omni wheels)."""

    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array([self.v_min, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def twist(self, action: np.ndarray) -> Twist:
        s = self.spec.max_speed
        return (
            float(action[0]) * s["linear"],
            float(action[1]) * s.get("lateral", s["linear"]),
            float(action[2]) * s["angular"],
        )

    def effort(self, action: np.ndarray) -> Tuple[float, float]:
        return float(np.hypot(action[0], action[1])), float(action[2])

    def discrete_actions(self) -> List[Sequence[float]]:
        return [
            [1, 0, 0],
            [1, 0, 0.5],
            [1, 0, -0.5],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]


DRIVES: Dict[str, Type[Drive]] = {"diff": DiffDrive, "omni": OmniDrive}


def make_drive(spec: RobotSpec) -> Drive:
    try:
        return DRIVES[spec.drive](spec)
    except KeyError:
        raise ValueError(
            f"Unknown drive {spec.drive!r}; known: {sorted(DRIVES)}"
        ) from None
