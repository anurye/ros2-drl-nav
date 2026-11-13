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

"""Algorithms for ``train``/``evaluate``: any SB3-compatible class plus a configs/algo file."""

from pathlib import Path
from typing import Dict, Type

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise

ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "dqn": DQN,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "ppo": PPO,
}


def make_model(
    algo: DictConfig, env: gym.Env, seed: int, log_dir: Path, device: str = "auto"
) -> BaseAlgorithm:
    """Instantiate the SB3 model described by a ``configs/algo/*.yaml`` entry."""
    cls = ALGOS[algo.name]
    params = OmegaConf.to_container(algo.params, resolve=True)
    policy = params.pop("policy", "MlpPolicy")
    if "action_noise_sigma" in algo:
        n = env.action_space.shape[0]
        params["action_noise"] = NormalActionNoise(
            mean=np.zeros(n), sigma=float(algo.action_noise_sigma) * np.ones(n)
        )
    return cls(
        policy,
        env,
        seed=seed,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir),
        **params,
    )


def load_model(name: str, path: Path, device: str = "cpu") -> BaseAlgorithm:
    return ALGOS[name].load(path, device=device)
