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

"""
Train an agent on DrlNav/TurtleBot4-v0 (Hydra app, see configs/train.yaml).

    ros2 run drl_nav train                              # SAC, defaults
    ros2 run drl_nav train algo=td3 seed=1              # overrides
    ros2 run drl_nav train -m algo=sac,tqc seed=0,1,2   # sweep

Outputs: experiments/<robot>/<tier>/<algo>/seed_<seed>/ (models, checkpoints, monitor.csv,
.hydra/config.yaml).
"""

from pathlib import Path
import random

from drl_nav.rl.common import config_dir, make_env
from drl_nav.rl.registry import make_model
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch

CONFIG_DIR = str(config_dir())


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def progress_bar_available() -> bool:
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return True


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))

    set_seeds(cfg.seed)
    env = make_env(
        cfg.env,
        cfg.launch_simulation,
        monitor_path=run_dir / "monitor.csv",
        discrete_actions=cfg.algo.get("discrete_actions"),
        robot_cfg=cfg.robot,
    )
    model = make_model(
        cfg.algo, env, seed=cfg.seed, log_dir=run_dir / "tb", device=cfg.device
    )
    callbacks = [
        CheckpointCallback(
            save_freq=cfg.save_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="model",
        ),
        EvalCallback(
            env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=cfg.eval_freq,
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
        ),
    ]
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            log_interval=cfg.log_interval,
            progress_bar=progress_bar_available(),
            tb_log_name=cfg.algo.name,
        )
    except KeyboardInterrupt:
        pass
    finally:
        model.save(run_dir / "final_model")
        env.close()


if __name__ == "__main__":
    main()
