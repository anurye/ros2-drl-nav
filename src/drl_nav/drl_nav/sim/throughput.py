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
Measure simulator/environment throughput with a random policy.

    ros2 run drl_nav sim_throughput --steps 300 --physics-step 0.002

Prints env steps per second and the real-time factor (simulated / wall seconds).
"""

import argparse
from pathlib import Path
import re
import tempfile
import time

from ament_index_python.packages import get_package_share_directory
import drl_nav
import gymnasium as gym


def world_with_step_size(world_name: str, step_size: float) -> Path:
    """Copy of the world file with a different <max_step_size>."""
    src = (
        Path(get_package_share_directory("drl_nav_worlds"))
        / "worlds"
        / f"{world_name}.sdf"
    )
    text = re.sub(
        r"<max_step_size>[^<]+</max_step_size>",
        f"<max_step_size>{step_size}</max_step_size>",
        src.read_text(),
    )
    out = Path(tempfile.mkdtemp()) / f"{world_name}.sdf"
    out.write_text(text)
    return out


def run(
    steps: int,
    physics_step: float,
    time_delta: float,
    robot: str = "turtlebot4",
    world_name: str = "static_world",
) -> dict:
    env = gym.make(
        drl_nav.env_id(robot),
        launch_simulation=True,
        launch_args={"world": str(world_with_step_size(world_name, physics_step))},
        physics_step_size=physics_step,
        time_delta=time_delta,
        goal_sampling_bias="uniform",
    )
    try:
        _, info = env.reset(seed=0)
        t_sim0, t_wall0 = info["sim_time"], time.perf_counter()
        episodes = 0
        for _ in range(steps):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                episodes += 1
                _, info = env.reset()
        wall = time.perf_counter() - t_wall0
        sim = info["sim_time"] - t_sim0
    finally:
        env.close()
    return {
        "robot": robot,
        "physics_step": physics_step,
        "time_delta": time_delta,
        "steps_per_s": steps / wall,
        "rtf": sim / wall,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--physics-step", type=float, default=0.004)
    parser.add_argument("--time-delta", type=float, default=0.1)
    parser.add_argument("--robot", default="turtlebot4")
    args = parser.parse_args()
    result = run(args.steps, args.physics_step, args.time_delta, args.robot)
    print(
        " ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in result.items()
        )
    )


if __name__ == "__main__":
    main()
