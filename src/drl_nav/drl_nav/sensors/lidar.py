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

"""Laser scan processing without ROS dependencies so that it is unit-testable."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ScanSpec:
    """Static properties of a laser scanner, taken from the first LaserScan message."""

    angle_min: float
    angle_max: float
    range_min: float
    range_max: float
    num_ranges: int

    @property
    def angle_increment(self) -> float:
        if self.num_ranges <= 1:
            return 0.0
        return (self.angle_max - self.angle_min) / (self.num_ranges - 1)


def bin_scan(
    ranges: Sequence[float],
    spec: ScanSpec,
    num_bins: int,
    yaw_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce a laser scan to the minimum range per angular sector.

    The field of view is split into ``num_bins`` equally wide sectors. For every sector the
    smallest valid reading and its bearing are returned. Readings that are NaN, infinite or
    outside ``[range_min, range_max]`` are ignored. A sector without a valid reading reports
    ``range_max`` at the sector centre. ``yaw_offset`` (lidar frame yaw in the robot frame) is
    added to the bearings, which are wrapped to ``[-pi, pi)``.

    Returns
    -------
    (min_ranges, min_ranges_angle): two float32 arrays of shape ``(num_bins,)``.

    """
    ranges = np.asarray(ranges, dtype=np.float64)
    num_ranges = ranges.size
    if num_ranges == 0:
        raise ValueError("Cannot bin an empty scan")

    num_bins = int(min(max(1, num_bins), num_ranges))
    span = spec.angle_max - spec.angle_min
    bin_width = span / num_bins

    angles = np.linspace(spec.angle_min, spec.angle_max, num_ranges)
    centres = spec.angle_min + (np.arange(num_bins) + 0.5) * bin_width

    min_ranges = np.full(num_bins, spec.range_max, dtype=np.float64)
    min_angles = centres.copy()

    valid = (
        np.isfinite(ranges) & (ranges >= spec.range_min) & (ranges <= spec.range_max)
    )
    if valid.any():
        if bin_width > 0:
            bin_idx = np.floor((angles - spec.angle_min) / bin_width).astype(int)
        else:
            bin_idx = np.zeros(num_ranges, dtype=int)
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)

        v_ranges = ranges[valid]
        v_angles = angles[valid]
        v_bins = bin_idx[valid]

        # Sort by (bin, range) so the first entry of every bin is its minimum.
        order = np.lexsort((v_ranges, v_bins))
        sorted_bins = v_bins[order]
        bins_present, first_idx = np.unique(sorted_bins, return_index=True)
        min_ranges[bins_present] = v_ranges[order][first_idx]
        min_angles[bins_present] = v_angles[order][first_idx]

    min_angles = (min_angles + yaw_offset + np.pi) % (2 * np.pi) - np.pi
    return min_ranges.astype(np.float32), min_angles.astype(np.float32)
