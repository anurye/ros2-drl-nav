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

"""Robot description for the environment: one YAML in ``configs/robot/`` per robot."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


@dataclass
class Topics:
    scan: str = "/scan"
    odom: str = "/ground_truth/odom"
    cmd_vel: str = "/cmd_vel"
    contacts: List[str] = field(default_factory=list)


@dataclass
class RobotSpec:
    """What the environment needs to know about a robot; everything else is in its ROS package."""

    name: str
    drive: str  # key of drl_nav.robots.drives.DRIVES
    package: str  # ROS package providing launch/spawn.launch.py
    model_name: str  # entity name in Gazebo
    max_speed: Dict[str, float]  # linear, angular[, lateral] in m/s, rad/s
    allow_reverse: bool = False
    footprint_radius: float = 0.3  # map inflation and reachability checks
    base_frame: str = "base_link"
    lidar_yaw: float = 0.0  # yaw of the lidar frame in base_frame [rad]
    startup_settle_time: float = 0.0  # sim seconds to run before the first pause
    topics: Topics = field(default_factory=Topics)
    tier_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RobotSpec":
        data = dict(data)
        data["topics"] = Topics(**data.get("topics", {}))
        return cls(**data)

    @classmethod
    def load(
        cls, name_or_path: Union[str, Path], config_dir: Optional[Path] = None
    ) -> "RobotSpec":
        path = Path(name_or_path)
        if path.suffix != ".yaml":
            from drl_nav.rl.common import config_dir as default_config_dir

            path = (
                (config_dir or default_config_dir()) / "robot" / f"{name_or_path}.yaml"
            )
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def resolve(robot: Union[str, RobotSpec, dict]) -> RobotSpec:
    if isinstance(robot, RobotSpec):
        return robot
    if isinstance(robot, dict):
        return RobotSpec.from_dict(robot)
    return RobotSpec.load(robot)
