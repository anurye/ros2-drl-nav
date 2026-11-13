# ros2-drl-nav

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![CI](https://github.com/anurye/ros2-drl-nav/actions/workflows/ros.yaml/badge.svg)](https://github.com/anurye/ros2-drl-nav/actions/workflows/ros.yaml)
[![ROS 2](https://img.shields.io/badge/ROS2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange.svg)](https://gazebosim.org/docs/harmonic/)

A framework for training and benchmarking deep reinforcement learning (DRL) policies for robot navigation using **ROS 2 Jazzy** and **Gazebo Harmonic**.

Currently supports the **TurtleBot4** differential-drive robot, with Ackermann-steered and omnidirectional robots planned.

## Leaderboard

Benchmark results for DQN, DDPG, TD3, SAC, and PPO.
<!-- leaderboard:start -->
Benchmark runs in progress.
<!-- leaderboard:end -->

## Getting started

### Installation

The recommended setup uses [Docker](https://docs.docker.com/engine/install/). Alternatively, open the repository in a VS Code dev container, which automatically builds the workspace.

```bash
git clone https://github.com/anurye/ros2-drl-nav.git
cd ros2-drl-nav

docker compose up -d dev
docker compose exec dev ./setup.sh
docker compose exec dev ./build.sh
docker compose exec dev bash
```

### Training and evaluation

Once inside the container, verify the environment and start training:

```bash
# Check the environment with a scripted policy
xvfb-run -a ros2 run drl_nav check_env --launch

# Train SAC on TurtleBot4 (medium difficulty)
xvfb-run -a ros2 run drl_nav train

# Train TD3 on the hard environment with seed 1
xvfb-run -a ros2 run drl_nav train env=hard algo=td3 seed=1
```

Training is configured with Hydra, so algorithms, environments, seeds, and other parameters can be changed through command-line overrides.

Evaluate a trained policy and update the leaderboard:

```bash
xvfb-run -a ros2 run drl_nav evaluate \
  experiments/turtlebot4/medium/sac/seed_42 \
  --launch --results-dir results

ros2 run drl_nav leaderboard
```

Publish a trained model to Hugging Face:

```bash
ros2 run drl_nav publish \
  experiments/turtlebot4/medium/sac/seed_42 \
  --repo-id <user>/tb4-nav-sac
```

Checkpoints, tb logs, and resolved configs are saved under:

`experiments/<robot>/<tier>/<algo>/seed_<seed>/`

Parallel training currently requires separate containers with distinct `ROS_DOMAIN_ID` and `GZ_PARTITION` values.

## Using the environment

Environments follow the standard Gymnasium API and can be used independently of the provided training pipeline.

```python
import gymnasium as gym
import drl_nav  # Registers DrlNav/* environments

env = gym.make("DrlNav/TurtleBot4-v0", launch_simulation=True)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)

env.close()
```

### Difficulty levels

Three difficulty levels are available, sharing the same world but differing in obstacle density, clearance, and start–goal distance.

| Environment ID | `env=` | Obstacles | Clearance | Start–goal path |
|---|---|---|---|---|
| `DrlNav/TurtleBot4Easy-v0` | `easy` | 0 | – | 1.5–5 m |
| `DrlNav/TurtleBot4-v0` | `medium` | 12 | 2.0 m | ≥ 1.5 m |
| `DrlNav/TurtleBot4Hard-v0` | `hard` | 20 | 1.2 m | ≥ 5 m |

Distances are measured along the shortest navigable path.

Environment parameters are documented in [NavigationEnv](https://github.com/anurye/ros2-drl-nav/blob/main/src/drl_nav/drl_nav/envs/navigation.py); robots, tiers and algorithms are configured under `configs/`.

## Running the simulation

The Gazebo simulation can be launched independently of the training environment.

```bash
# Gazebo GUI + RViz
ros2 launch drl_nav simulation.launch.py robot:=turtlebot4

# Headless simulation
xvfb-run -a ros2 launch drl_nav simulation.launch.py \
  headless:=True use_rviz:=False

# Check simulator services and stepping
ros2 run drl_nav check_sim

# Measure simulation throughput
ros2 run drl_nav sim_throughput
```

### GUI access

On systems without a host X server, the simulation GUI can be accessed through noVNC.

```bash
./scripts/novnc.sh
```

Open [http://localhost:6080/vnc.html](http://localhost:6080/vnc.html) and launch the simulation with `DISPLAY=:1`.

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
