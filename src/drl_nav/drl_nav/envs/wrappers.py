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

"""Gymnasium wrappers for the navigation environment."""

from typing import Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteActions(gym.ActionWrapper):
    """Map a discrete index to one of a fixed set of (linear, angular) continuous actions."""

    def __init__(self, env: gym.Env, actions: Sequence[Tuple[float, float]]):
        super().__init__(env)
        if not actions:
            raise ValueError("actions must not be empty")
        self._actions = np.asarray(actions, dtype=np.float32)
        low, high = env.action_space.low, env.action_space.high
        if np.any(self._actions < low) or np.any(self._actions > high):
            raise ValueError(f"actions must lie within {low} .. {high}")
        self.action_space = spaces.Discrete(len(self._actions))

    def action(self, action: int) -> np.ndarray:
        return self._actions[int(action)]
