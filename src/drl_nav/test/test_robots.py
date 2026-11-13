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

from pathlib import Path

from drl_nav import env_id, ROBOTS, TIERS
from drl_nav.robots.drives import DRIVES, make_drive
from drl_nav.robots.spec import RobotSpec
import numpy as np
import pytest

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@pytest.mark.parametrize("robot", ROBOTS)
def test_robot_configs_load(robot):
    spec = RobotSpec.load(robot, config_dir=CONFIGS)
    assert spec.name == robot and spec.drive in DRIVES
    assert {"linear", "angular"} <= spec.max_speed.keys()


def test_env_ids():
    assert env_id("turtlebot4") == "DrlNav/TurtleBot4-v0"
    assert env_id("turtlebot4", "Hard") == "DrlNav/TurtleBot4Hard-v0"
    assert set(TIERS) == {"Easy", "", "Hard"}


def spec(drive, **kwargs):
    base = {
        "name": "x",
        "drive": drive,
        "package": "p",
        "model_name": "x",
        "max_speed": {"linear": 0.5, "angular": 1.0, "lateral": 0.4},
    }
    base.update(kwargs)
    return RobotSpec.from_dict(base)


def test_diff_drive_twist_and_space():
    d = make_drive(spec("diff"))
    assert d.action_space().shape == (2,)
    assert d.action_space().low[0] == 0.0
    assert d.twist(np.array([1.0, -0.5])) == (0.5, 0.0, -0.5)
    assert d.effort(np.array([1.0, -0.5])) == (1.0, -0.5)
    assert all(len(a) == 2 for a in d.discrete_actions())


def test_reverse_widens_linear_range():
    assert make_drive(spec("diff", allow_reverse=True)).action_space().low[0] == -1.0


def test_omni_drive_twist_and_space():
    d = make_drive(spec("omni"))
    assert d.action_space().shape == (3,)
    assert d.twist(np.array([1.0, 0.5, -1.0])) == (0.5, 0.2, -1.0)
    speed, turn = d.effort(np.array([0.6, 0.8, 0.3]))
    assert speed == pytest.approx(1.0) and turn == pytest.approx(0.3)
    assert all(len(a) == 3 for a in d.discrete_actions())


def test_unknown_drive_rejected():
    with pytest.raises(ValueError):
        make_drive(spec("legs"))
