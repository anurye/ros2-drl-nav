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
Evaluate a training run, or a scripted baseline, on a fixed set of seeded scenarios.

    ros2 run drl_nav evaluate experiments/turtlebot4/medium/sac/seed_42 [--model best_model.zip]
    ros2 run drl_nav evaluate --baseline random

Scenario i is env.reset(seed=first_seed + i). Writes <run_dir>/eval.json (or --output) and,
with --results-dir, a copy named <robot>_<tier>_<algo>_seed<seed>.json for the leaderboard.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from typing import Callable, Dict, List

from drl_nav.rl.common import config_dir, load_run_config, make_env
from drl_nav.rl.registry import load_model
import gymnasium as gym
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import DictConfig, OmegaConf

Policy = Callable[[np.ndarray], np.ndarray]


def choices(group: str):
    return sorted(
        p.stem
        for p in (config_dir() / group).glob("*.yaml")
        if not p.stem.startswith("_")
    )


def run_episode(env: gym.Env, policy: Policy, seed: int) -> Dict:
    obs, info = env.reset(seed=seed)
    shortest = info["shortest_path_m"]
    path, prev = 0.0, info["robot_pose"][:2]
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        x, y, _ = info["robot_pose"]
        path += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
        if terminated or truncated:
            break
    success = bool(info["is_success"])
    return {
        "seed": seed,
        "success": success,
        "collision": bool(info["collision"]),
        "timeout": bool(truncated and not terminated),
        "steps": info["episode_steps"],
        "shortest_path_m": shortest,
        "path_m": path,
        "spl": shortest / max(path, shortest) if success else 0.0,
    }


def summarise(episodes: List[Dict]) -> Dict:
    n = len(episodes)
    steps_success = [e["steps"] for e in episodes if e["success"]]
    return {
        "episodes": n,
        "success_rate": sum(e["success"] for e in episodes) / n,
        "collision_rate": sum(e["collision"] for e in episodes) / n,
        "timeout_rate": sum(e["timeout"] for e in episodes) / n,
        "spl": sum(e["spl"] for e in episodes) / n,
        "mean_steps_to_goal": statistics.mean(steps_success) if steps_success else None,
    }


def evaluate(
    env: gym.Env, policy: Policy, episodes: int, first_seed: int
) -> List[Dict]:
    results = []
    for i in range(episodes):
        results.append(run_episode(env, policy, first_seed + i))
        e = results[-1]
        print(
            f"[{i + 1:3d}/{episodes}] seed={e['seed']} success={e['success']} "
            f"collision={e['collision']} steps={e['steps']}",
            file=sys.stderr,
        )
    return results


def compose_config(robot: str, tier: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        return compose("train", overrides=[f"robot={robot}", f"env={tier}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("run_dir", nargs="?", type=Path, help="training run directory")
    parser.add_argument(
        "--model", default="final_model.zip", help="model file inside run_dir"
    )
    parser.add_argument("--baseline", choices=["random", "straight"])
    parser.add_argument(
        "--robot",
        choices=choices("robot"),
        default="turtlebot4",
        help="robot for baselines (runs use their .hydra/config.yaml)",
    )
    parser.add_argument(
        "--env", choices=choices("env"), default="medium", help="tier for baselines"
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=10_000)
    parser.add_argument(
        "--launch", action="store_true", help="start a headless simulation"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    if (args.run_dir is None) == (args.baseline is None):
        parser.error("give either run_dir or --baseline")

    discrete = None
    if args.run_dir is not None:
        cfg = load_run_config(args.run_dir)
        name, seed = cfg.algo.name, cfg.seed
        discrete = cfg.algo.get("discrete_actions")
    else:
        cfg, name, seed = compose_config(args.robot, args.env), args.baseline, None
    env_cfg = cfg.env

    env = make_env(
        env_cfg,
        launch_simulation=args.launch,
        discrete_actions=discrete,
        robot_cfg=cfg.robot,
    )
    try:
        if args.baseline == "random":
            policy = lambda obs: env.action_space.sample()  # noqa: E731
        elif args.baseline == "straight":
            policy = lambda obs: np.array([1.0, 0.0], dtype=np.float32)  # noqa: E731
        else:
            model = load_model(name, args.run_dir / args.model)
            policy = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
        episodes = evaluate(env, policy, args.episodes, args.first_seed)
    finally:
        env.close()

    summary = summarise(episodes)
    report = {
        "algo": name,
        "train_seed": seed,
        "model": None if args.baseline else str(args.run_dir / args.model),
        "robot": OmegaConf.to_container(cfg.robot, resolve=True),
        "env": OmegaConf.to_container(env_cfg, resolve=True),
        "eval": {"episodes": args.episodes, "first_seed": args.first_seed},
        "summary": summary,
        "episodes": episodes,
    }
    default = args.run_dir / "eval.json" if args.run_dir else Path(f"{name}_eval.json")
    output = args.output or default
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}", file=sys.stderr)
    if args.results_dir:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{name}_seed{seed}" if seed is not None else name
        tag = f"{cfg.robot.name}_{env_cfg.name}_{tag}"
        shutil.copy(output, args.results_dir / f"{tag}.json")


if __name__ == "__main__":
    main()
