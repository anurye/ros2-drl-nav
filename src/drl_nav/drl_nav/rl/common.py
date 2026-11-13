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

"""Shared helpers for the RL scripts."""

from pathlib import Path
from typing import Optional, Sequence, Union

from ament_index_python.packages import get_package_share_directory
import drl_nav  # noqa: F401  registers the DrlNav/* ids
from drl_nav.envs.wrappers import DiscreteActions
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.monitor import Monitor

RUN_CONFIG = Path(".hydra") / "config.yaml"


def config_dir() -> Path:
    """Installed Hydra config directory, or the repository's configs/ when run from source."""
    try:
        installed = Path(get_package_share_directory("drl_nav")) / "configs"
        if installed.exists():
            return installed
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[4] / "configs"


def make_env(
    env_cfg: DictConfig,
    launch_simulation: bool,
    monitor_path: Optional[Path] = None,
    discrete_actions: Union[None, str, Sequence[Sequence[float]]] = None,
    robot_cfg: Optional[DictConfig] = None,
) -> gym.Env:
    """
    ``gym.make`` from ``configs/env`` and ``configs/robot`` entries, flattened for MLP policies.

    ``discrete_actions``: a list of actions, or ``'drive'`` for the drive's default set (DQN).
    """
    kwargs = OmegaConf.to_container(env_cfg, resolve=True)
    env_id = kwargs.pop("id")
    kwargs.pop("name", None)
    if robot_cfg is not None:
        kwargs["robot"] = OmegaConf.to_container(robot_cfg, resolve=True)
    env = gym.make(env_id, launch_simulation=launch_simulation, **kwargs)
    env = gym.wrappers.FlattenObservation(env)
    if discrete_actions == "drive":
        discrete_actions = env.unwrapped.drive.discrete_actions()
    if discrete_actions:
        env = DiscreteActions(env, [tuple(a) for a in discrete_actions])
    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
    return Monitor(
        env,
        filename=None if monitor_path is None else str(monitor_path),
        info_keywords=("is_success", "collision"),
    )


def load_run_config(run_dir: Path) -> DictConfig:
    """Load the composed Hydra config of a training run."""
    path = run_dir / RUN_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; is {run_dir} a training run directory?"
        )
    return OmegaConf.load(path)
