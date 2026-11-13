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
Upload an evaluated run to the Hugging Face Hub with a generated model card.

    ros2 run drl_nav publish experiments/turtlebot4/medium/sac/seed_42 --repo-id <user>/tb4-nav-sac

Needs ``huggingface-cli login`` (or HF_TOKEN) and an ``eval.json`` from ``evaluate``.
Uploads final_model.zip, the run config, eval.json and README.md; records the URL in eval.json
so the leaderboard can link to it. Load it back with
``huggingface_sb3.load_from_hub(repo_id, 'final_model.zip')``.
"""

import argparse
import json
from pathlib import Path
import sys

from drl_nav.rl.common import load_run_config, RUN_CONFIG
from huggingface_hub import HfApi
from omegaconf import OmegaConf

CARD = """---
library_name: stable-baselines3
tags:
- ros2-drl-nav
- reinforcement-learning
- robotics
- ros2
- gazebo
- {algo}
model-index:
- name: {algo} {env_id}
  results:
  - task: {{type: reinforcement-learning, name: point-goal navigation}}
    dataset: {{type: {env_id}, name: {env_id}}}
    metrics:
    - {{type: success_rate, value: {success:.3f}, name: success rate}}
    - {{type: collision_rate, value: {collision:.3f}, name: collision rate}}
    - {{type: spl, value: {spl:.3f}, name: SPL}}
---

# {algo_upper} on {env_id}

Point-goal navigation with a {robot} in ROS 2 Jazzy / Gazebo Harmonic, trained with
[drl_nav](https://github.com/anurye/drl_nav) and Stable-Baselines3.

| Success | Collision | Timeout | SPL | Steps to goal |
|---|---|---|---|---|
| {success:.1%} | {collision:.1%} | {timeout:.1%} | {spl:.2f} | {steps} |

Evaluated on {episodes} fixed scenarios (`evaluate`, seeds from {first_seed}).
Training seed {seed}, {timesteps} environment steps.

## Usage

```python
from huggingface_sb3 import load_from_hub
from drl_nav.rl.registry import load_model

path = load_from_hub('{repo_id}', 'final_model.zip')
model = load_model('{algo}', path)
```

The observation is the flattened `Dict` observation of `{env_id}` with the environment
settings in `config.yaml` (notably `num_bins`). See the repository README for running the
simulation.

## Configuration

```yaml
{config_yaml}```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--repo-id", required=True, help="e.g. <user>/ros2-drl-nav-medium-sac"
    )
    parser.add_argument("--model", default="final_model.zip")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="write the card, upload nothing"
    )
    args = parser.parse_args()

    cfg = load_run_config(args.run_dir)
    eval_path = args.run_dir / "eval.json"
    if not eval_path.exists():
        raise SystemExit(f"{eval_path} missing: run `evaluate {args.run_dir}` first")
    report = json.loads(eval_path.read_text())
    s = report["summary"]
    steps = s["mean_steps_to_goal"]
    card = CARD.format(
        algo=cfg.algo.name,
        algo_upper=cfg.algo.name.upper(),
        repo_id=args.repo_id,
        env_id=cfg.env.id,
        robot=cfg.robot.name,
        success=s["success_rate"],
        collision=s["collision_rate"],
        timeout=s["timeout_rate"],
        spl=s["spl"],
        steps="–" if steps is None else f"{steps:.0f}",
        episodes=report["eval"]["episodes"],
        first_seed=report["eval"]["first_seed"],
        seed=cfg.seed,
        timesteps=cfg.total_timesteps,
        config_yaml=OmegaConf.to_yaml(cfg),
    )
    card_path = args.run_dir / "README.md"
    card_path.write_text(card)
    print(f"wrote {card_path}", file=sys.stderr)
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    for src, dst in [
        (args.run_dir / args.model, "final_model.zip"),
        (args.run_dir / RUN_CONFIG, "config.yaml"),
        (eval_path, "eval.json"),
        (card_path, "README.md"),
    ]:
        api.upload_file(
            path_or_fileobj=str(src), path_in_repo=dst, repo_id=args.repo_id
        )
    url = f"https://huggingface.co/{args.repo_id}"
    report["hub_url"] = url
    eval_path.write_text(json.dumps(report, indent=2))
    print(url)


if __name__ == "__main__":
    main()
