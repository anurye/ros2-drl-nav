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

"""Reward shaping for the point-goal navigation task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    Weights of the reward terms.

    With the defaults the reward is:

    * ``goal_reward`` when the goal is reached,
    * ``collision_penalty`` on collision,
    * otherwise ``progress_weight * progress + linear_weight * v - angular_weight * |w|
      - step_penalty`` plus ``obstacle_weight * (min_range - obstacle_distance_threshold)``
      when the closest obstacle is nearer than ``obstacle_distance_threshold``.

    ``progress`` is the decrease of the goal distance during the step [m]; ``v`` and ``w`` are
    the normalised actions in ``[0, 1]`` and ``[-1, 1]``.
    """

    goal_reward: float = 100.0
    collision_penalty: float = -100.0
    progress_weight: float = 10.0
    linear_weight: float = 0.5
    angular_weight: float = 0.5
    step_penalty: float = 0.001
    obstacle_distance_threshold: float = 1.0
    obstacle_weight: float = 0.5


def compute_reward(
    linear_action: float,
    angular_action: float,
    min_range: float,
    progress: float,
    goal_reached: bool,
    collision: bool,
    config: RewardConfig = RewardConfig(),
) -> float:
    if goal_reached:
        return float(config.goal_reward)
    if collision:
        return float(config.collision_penalty)

    obstacle_reward = 0.0
    if min_range < config.obstacle_distance_threshold:
        obstacle_reward = config.obstacle_weight * (
            min_range - config.obstacle_distance_threshold
        )

    action_reward = (
        config.progress_weight * progress
        + config.linear_weight * linear_action
        - config.angular_weight * abs(angular_action)
        - config.step_penalty
    )
    return float(obstacle_reward + action_reward)
