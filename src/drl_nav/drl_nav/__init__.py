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


from gymnasium.envs.registration import register

# Difficulty tiers: same world, different scenario distribution. Distances are shortest-path
# lengths on the map, so goals behind walls count as far. Unused obstacle models are parked.
TIERS = {
    "Easy": {"num_obstacles": 0, "min_separation": 1.5, "max_separation": 5.0},
    "": {"num_obstacles": 12, "obstacle_clearance": 2.0, "min_separation": 1.5},
    "Hard": {"num_obstacles": 20, "obstacle_clearance": 1.2, "min_separation": 5.0},
}

# Robots: configs/robot/<name>.yaml and a ROS package drl_nav_<name>.
ROBOTS = {"turtlebot4": "TurtleBot4"}


def env_id(robot: str, tier: str = "") -> str:
    return f"DrlNav/{ROBOTS[robot]}{tier}-v0"


for _robot in ROBOTS:
    for _tier, _kwargs in TIERS.items():
        register(
            id=env_id(_robot, _tier),
            entry_point="drl_nav.envs.navigation:NavigationEnv",
            max_episode_steps=500,
            kwargs={"robot": _robot, **_kwargs},
        )
