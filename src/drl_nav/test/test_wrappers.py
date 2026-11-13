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

from drl_nav.envs.wrappers import DiscreteActions
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest


class _ContinuousEnv(gym.Env):
    action_space = spaces.Box(
        np.array([0.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32
    )
    observation_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return (
            np.asarray(action[:1], dtype=np.float32),
            0.0,
            False,
            False,
            {"action": action},
        )


def test_discrete_indices_map_to_continuous_actions():
    env = DiscreteActions(_ContinuousEnv(), [(1.0, 0.0), (0.0, 1.0)])
    assert env.action_space == spaces.Discrete(2)
    _, _, _, _, info = env.step(1)
    assert info["action"] == pytest.approx([0.0, 1.0])


def test_rejects_actions_outside_bounds():
    with pytest.raises(ValueError):
        DiscreteActions(_ContinuousEnv(), [(2.0, 0.0)])
