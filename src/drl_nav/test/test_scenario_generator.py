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

import math
from pathlib import Path

from drl_nav import TIERS
from drl_nav.envs.scenarios import ScenarioGenerator
import numpy as np
import pytest

MAPS = Path(__file__).resolve().parents[2] / "worlds" / "drl_nav_worlds" / "maps"


def make_generator(seed=None, rng=None, **kwargs):
    params = {"robot_radius": 0.4, "min_separation": 1.5, "obstacle_clearance": 1.0}
    params.update(kwargs)
    return ScenarioGenerator(
        map_path=MAPS / "static_world.pgm",
        yaml_path=MAPS / "static_world.yaml",
        seed=seed,
        rng=rng,
        **params,
    )


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_same_seed_gives_same_scenario():
    a, b = make_generator(seed=7), make_generator(seed=7)
    assert a.generate_start_goal() == b.generate_start_goal()
    start, goal = a.generate_start_goal()
    b.generate_start_goal()
    assert a.generate_obstacles(5, start, goal) == b.generate_obstacles(5, start, goal)


def test_external_generator_is_used_and_can_be_replaced():
    gen = make_generator(rng=np.random.default_rng(1))
    first = gen.generate_start_goal()
    gen.reseed(np.random.default_rng(1))
    assert gen.generate_start_goal() == first
    gen.reseed(2)
    assert gen.generate_start_goal() != first


def test_walls_are_inflated_by_robot_radius():
    # Arena interior spans ±5.925 m; inflation of 0.4 m leaves |x| < 5.525.
    gen = make_generator(seed=0)
    xs = [gen.map_to_world(c)[0] for c in gen._cells]
    assert 5.4 < max(xs) < 5.6
    assert gen.free[gen.world_to_map((5.7, 2.0))] == 0
    assert gen.free[gen.world_to_map((5.3, 2.0))] == 1


def test_map_orientation_row_zero_is_top():
    gen = make_generator(seed=0)
    assert gen.map_to_world((0, 0))[1] > gen.map_to_world((gen.free.shape[0] - 1, 0))[1]
    for pos in [(0.0, 0.0), (3.2, -4.1), (-5.0, 5.0)]:
        back = gen.map_to_world(gen.world_to_map(pos))
        assert dist(back, pos) < gen.resolution


def test_start_goal_respect_path_length_limits():
    gen = make_generator(seed=0, min_separation=3.0, max_separation=6.0)
    for _ in range(20):
        start, goal = gen.generate_start_goal()
        length = gen.shortest_path_length(start, goal)
        assert 3.0 - 1e-9 <= length <= 6.0 + 1e-9
        assert length >= dist(start, goal) - 1e-9
        assert gen.free[gen.world_to_map(start)] and gen.free[gen.world_to_map(goal)]


def test_path_through_wall_is_longer_than_straight_line():
    # Opposite rooms: the only way round is through the central opening.
    gen = make_generator(seed=0)
    start, goal = (4.0, 0.5), (4.0, -0.5)
    assert dist(start, goal) == pytest.approx(1.0)
    assert gen.shortest_path_length(start, goal) > 5.0


def test_obstacles_keep_clearance_and_goal_reachable():
    clearance = 1.0
    gen = make_generator(seed=3, obstacle_clearance=clearance)
    start, goal = gen.generate_start_goal()
    obstacles = gen.generate_obstacles(12, start, goal)
    assert 0 < len(obstacles) <= 12
    margin = clearance - gen.resolution
    for i, o in enumerate(obstacles):
        assert dist(o, start) > margin and dist(o, goal) > margin
        assert all(dist(o, p) > margin for p in obstacles[i + 1 :])
    assert math.isfinite(gen.shortest_path_length(start, goal, obstacles))


def test_blocked_opening_is_detected():
    gen = make_generator(seed=0)
    start, goal = (-3.0, 3.0), (3.0, -3.0)
    wall = [(0.0, y) for y in np.arange(-2.0, 2.1, 0.5)]
    assert math.isinf(gen.shortest_path_length(start, goal, wall))
    assert math.isfinite(gen.shortest_path_length(start, goal, wall[:3]))


def test_no_obstacles_requested():
    gen = make_generator(seed=0)
    assert gen.generate_obstacles(0, (0.0, 0.0), (2.0, 0.0)) == []


@pytest.mark.parametrize("env_id, kwargs", TIERS.items())
def test_tiers_are_feasible(env_id, kwargs):
    params = {k: v for k, v in kwargs.items() if k != "num_obstacles"}
    gen = make_generator(seed=1, **params)
    for _ in range(5):
        start, goal = gen.generate_start_goal()
        obstacles = gen.generate_obstacles(kwargs["num_obstacles"], start, goal)
        assert len(obstacles) == kwargs["num_obstacles"]
        length = gen.shortest_path_length(start, goal, obstacles)
        assert math.isfinite(length) and length >= kwargs["min_separation"]
        if "max_separation" in kwargs:
            assert length <= kwargs["max_separation"] + 1e-9


def test_world_map_round_trip():
    gen = make_generator(seed=0)
    for cell in [(10, 10), (100, 37), (0, 0)]:
        assert gen.world_to_map(gen.map_to_world(cell)) == cell


@pytest.mark.parametrize("bias", ["uniform", "close", "far"])
def test_goal_sampling_bias_accepted(bias):
    make_generator(seed=0).generate_start_goal(goal_sampling_bias=bias)


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        make_generator(seed=0).generate_start_goal(goal_sampling_bias="nope")
    with pytest.raises(ValueError):
        make_generator(min_separation=3.0, max_separation=2.0)
