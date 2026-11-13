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

"""Start/goal/obstacle sampling on an occupancy map, with grid shortest paths."""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
import yaml

Point = Tuple[float, float]
Cell = Tuple[int, int]

_SQRT2 = float(np.sqrt(2.0))


def _disk(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1,) * 2)


def _grid_graph(free: np.ndarray) -> Tuple[np.ndarray, np.ndarray, csr_matrix]:
    """8-connected graph over the free cells; edge weights in cells."""
    cells = np.argwhere(free)
    index = -np.ones(free.shape, dtype=np.int64)
    index[free] = np.arange(len(cells))
    rows, cols, weights = [], [], []
    for dr, dc, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, _SQRT2), (1, -1, _SQRT2)):
        nbr = cells + (dr, dc)
        inside = (
            (nbr[:, 0] >= 0)
            & (nbr[:, 0] < free.shape[0])
            & (nbr[:, 1] >= 0)
            & (nbr[:, 1] < free.shape[1])
        )
        j = index[nbr[inside, 0], nbr[inside, 1]]
        i = index[cells[inside, 0], cells[inside, 1]]
        rows.append(i[j >= 0])
        cols.append(j[j >= 0])
        weights.append(np.full(int((j >= 0).sum()), w))
    n = len(cells)
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    return cells, index, graph


class ScenarioGenerator:
    """
    Sample navigation scenarios from an occupancy map (``map_server`` PGM + YAML).

    Walls are inflated by ``robot_radius``. Start-goal distances are shortest-path lengths on
    the inflated grid, so a goal behind a wall counts as far. Obstacles (cylinders of
    ``obstacle_radius``) keep ``obstacle_clearance`` from start, goal and each other and never
    disconnect the goal from the start.
    """

    def __init__(
        self,
        map_path: Path,
        yaml_path: Path,
        robot_radius: float = 0.3,
        min_separation: float = 1.5,
        max_separation: Optional[float] = None,
        obstacle_radius: float = 0.25,
        obstacle_clearance: float = 1.5,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not map_path.exists():
            raise FileNotFoundError(f"Map file {map_path} not found")
        if not yaml_path.exists():
            raise FileNotFoundError(f"Map metadata file {yaml_path} not found")
        if min(robot_radius, min_separation, obstacle_radius, obstacle_clearance) <= 0:
            raise ValueError("radii, separation and clearance must be positive")
        if max_separation is not None and max_separation < min_separation:
            raise ValueError("max_separation must be >= min_separation")

        self.map_path = map_path
        self.yaml_path = yaml_path
        self.robot_radius = robot_radius
        self.min_separation = min_separation
        self.max_separation = max_separation
        self.obstacle_radius = obstacle_radius
        self.obstacle_clearance = obstacle_clearance
        self.seed = seed
        self._rng = rng if rng is not None else np.random.default_rng(seed)

        self.metadata = self._load_metadata()
        self.resolution = float(self.metadata["resolution"])
        self.origin = tuple(float(v) for v in self.metadata["origin"])
        self.map_img, self.free = self._load_map()
        self._cells, self._index, self._graph = _grid_graph(self.free)
        # Obstacle centres keep their own radius from the inflated walls.
        self.obstacle_cells = np.argwhere(
            cv2.erode(self.free.astype(np.uint8), _disk(self._px(obstacle_radius))) == 1
        )

    def reseed(self, rng_or_seed: Union[int, np.random.Generator, None]) -> None:
        """Replace the random generator (pass ``env.np_random`` to follow Gymnasium seeding)."""
        if isinstance(rng_or_seed, np.random.Generator):
            self._rng = rng_or_seed
        else:
            self.seed = rng_or_seed
            self._rng = np.random.default_rng(rng_or_seed)

    def _load_metadata(self) -> dict:
        with open(self.yaml_path) as f:
            metadata = yaml.safe_load(f)
        missing = {"resolution", "origin", "free_thresh"} - metadata.keys()
        if missing:
            raise ValueError(f"Missing required YAML keys: {missing}")
        if len(metadata["origin"]) != 3:
            raise ValueError("origin must be [x, y, yaw]")
        return metadata

    def _load_map(self) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(self.map_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load map image from {self.map_path}")
        occupancy = img / 255.0 if self.metadata.get("negate") else (255 - img) / 255.0
        free = (occupancy < self.metadata["free_thresh"]).astype(np.uint8)
        free = cv2.erode(free, _disk(self._px(self.robot_radius))) == 1
        if not free.any():
            raise ValueError("No free space left after inflating walls by robot_radius")
        return img, free

    def _px(self, metres: float) -> int:
        return int(np.ceil(metres / self.resolution))

    def generate_start_goal(
        self,
        max_attempts: int = 100,
        goal_sampling_bias: str = "uniform",
        eps: float = 1e-5,
    ) -> Tuple[Point, Point]:
        """Start and goal whose path length lies in [min_separation, max_separation]."""
        if goal_sampling_bias not in ("uniform", "close", "far"):
            raise ValueError(f"Invalid goal_sampling_bias: {goal_sampling_bias}")
        lo = self.min_separation / self.resolution
        hi = (
            np.inf
            if self.max_separation is None
            else self.max_separation / self.resolution
        )

        for _ in range(max_attempts):
            start = self._rng.integers(len(self._cells))
            dist = dijkstra(self._graph, directed=False, indices=start)
            candidates = np.flatnonzero((dist >= lo) & (dist <= hi))
            if candidates.size == 0:
                continue
            if goal_sampling_bias == "uniform":
                weights = np.ones(candidates.size)
            elif goal_sampling_bias == "close":
                weights = 1.0 / (dist[candidates] + eps)
            else:
                weights = dist[candidates]
            goal = self._rng.choice(candidates, p=weights / weights.sum())
            return self.map_to_world(self._cells[start]), self.map_to_world(
                self._cells[goal]
            )

        raise RuntimeError(
            f"No start/goal pair within the separation limits "
            f"after {max_attempts} attempts"
        )

    def generate_obstacles(
        self,
        num_obstacles: int,
        start_pos: Point,
        goal_pos: Point,
        max_attempts: int = 20,
    ) -> List[Point]:
        """Obstacle centres with clearance from start/goal/each other; goal stays reachable."""
        if num_obstacles <= 0:
            return []
        clear_px = self.obstacle_clearance / self.resolution
        start = np.array(self.world_to_map(start_pos))
        goal = np.array(self.world_to_map(goal_pos))
        cells = self.obstacle_cells
        candidates = cells[
            (np.linalg.norm(cells - start, axis=1) > clear_px)
            & (np.linalg.norm(cells - goal, axis=1) > clear_px)
        ]
        reachable = np.isfinite(self.shortest_path_length(start_pos, goal_pos))
        if not reachable:
            self.logger.warning("Goal unreachable from start even without obstacles.")

        for _ in range(max_attempts):
            chosen: List[np.ndarray] = []
            remaining = candidates
            while len(chosen) < num_obstacles and len(remaining):
                cell = remaining[self._rng.integers(len(remaining))]
                chosen.append(cell)
                remaining = remaining[
                    np.linalg.norm(remaining - cell, axis=1) > clear_px
                ]
            positions = [self.map_to_world(c) for c in chosen]
            if not reachable or np.isfinite(
                self.shortest_path_length(start_pos, goal_pos, positions)
            ):
                if len(positions) < num_obstacles:
                    self.logger.warning(
                        f"Placed {len(positions)} of {num_obstacles} obstacles; "
                        "lower obstacle_clearance to fit more."
                    )
                return positions

        raise RuntimeError(f"Obstacles blocked the goal in {max_attempts} attempts")

    def shortest_path_length(
        self, start_pos: Point, goal_pos: Point, obstacles: Sequence[Point] = ()
    ) -> float:
        """Grid shortest path [m] on the inflated map with obstacles painted in; inf if blocked."""
        if len(obstacles):
            free = self.free.astype(np.uint8)
            r = self._px(self.obstacle_radius + self.robot_radius)
            for pos in obstacles:
                row, col = self.world_to_map(pos)
                cv2.circle(free, (col, row), r, 0, thickness=-1)
            cells, index, graph = _grid_graph(free == 1)
        else:
            index, graph = self._index, self._graph
        s, g = index[self.world_to_map(start_pos)], index[self.world_to_map(goal_pos)]
        if s < 0 or g < 0:
            return float("inf")
        return float(
            dijkstra(graph, directed=False, indices=s, min_only=True)[g]
            * self.resolution
        )

    def map_to_world(self, cell: Cell) -> Point:
        """(row, col) -> (x, y) of the cell centre; row 0 is the top of the image."""
        ox, oy, yaw = self.origin
        x_map = (cell[1] + 0.5) * self.resolution
        y_map = (self.free.shape[0] - cell[0] - 0.5) * self.resolution
        return (
            ox + x_map * np.cos(yaw) - y_map * np.sin(yaw),
            oy + x_map * np.sin(yaw) + y_map * np.cos(yaw),
        )

    def world_to_map(self, pos: Point) -> Cell:
        ox, oy, yaw = self.origin
        dx, dy = pos[0] - ox, pos[1] - oy
        x_map = dx * np.cos(yaw) + dy * np.sin(yaw)
        y_map = -dx * np.sin(yaw) + dy * np.cos(yaw)
        h, w = self.free.shape
        row = int(np.clip(h - y_map / self.resolution, 0, h - 1))
        col = int(np.clip(x_map / self.resolution, 0, w - 1))
        return row, col

    def plot_debug(
        self,
        start_pos: Optional[Point] = None,
        goal_pos: Optional[Point] = None,
        obstacles: Sequence[Point] = (),
    ) -> None:
        import matplotlib.pyplot as plt

        h, w = self.free.shape
        (x0, y0), (x1, y1) = (
            self.map_to_world((h - 1, 0)),
            self.map_to_world((0, w - 1)),
        )
        extent = [x0, x1, y0, y1]
        plt.figure(figsize=(8, 8))
        plt.imshow(self.map_img, cmap="gray", extent=extent)
        plt.imshow(
            np.ma.masked_where(~self.free, self.free),
            cmap="Greens",
            alpha=0.3,
            extent=extent,
        )
        for ox, oy in obstacles:
            plt.gca().add_patch(
                plt.Circle((ox, oy), self.obstacle_radius, color="tab:red")
            )
        if start_pos is not None:
            plt.plot(*start_pos, "D", color="lime", ms=12, mec="k", label="start")
        if goal_pos is not None:
            plt.plot(*goal_pos, "*", color="gold", ms=18, mec="k", label="goal")
        if start_pos is not None and goal_pos is not None:
            length = self.shortest_path_length(start_pos, goal_pos, obstacles)
            plt.title(f"shortest path {length:.2f} m")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


def main() -> None:
    """Sample one scenario on a map and plot it."""
    from ament_index_python.packages import get_package_share_directory

    maps = Path(get_package_share_directory("drl_nav_worlds")) / "maps"
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("-m", "--map", type=Path, default=maps / "static_world.pgm")
    parser.add_argument("-y", "--yaml", type=Path, default=maps / "static_world.yaml")
    parser.add_argument("--obstacles", type=int, default=12)
    parser.add_argument("--clearance", type=float, default=2.0)
    parser.add_argument("--min-separation", type=float, default=1.5)
    parser.add_argument("--max-separation", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    gen = ScenarioGenerator(
        args.map,
        args.yaml,
        robot_radius=0.4,
        min_separation=args.min_separation,
        max_separation=args.max_separation,
        obstacle_clearance=args.clearance,
        seed=args.seed,
    )
    start, goal = gen.generate_start_goal()
    obstacles = gen.generate_obstacles(args.obstacles, start, goal)
    print(f"start={start} goal={goal} obstacles={obstacles}")
    gen.plot_debug(start, goal, obstacles)


if __name__ == "__main__":
    main()
