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

from drl_nav.sensors.lidar import bin_scan, ScanSpec
import numpy as np
import pytest

SPEC = ScanSpec(
    angle_min=0.0, angle_max=2 * math.pi, range_min=0.16, range_max=20.0, num_ranges=360
)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def test_empty_sectors_report_range_max_at_sector_centre():
    ranges = np.full(360, np.inf)
    min_ranges, angles = bin_scan(ranges, SPEC, num_bins=4)
    assert min_ranges.shape == angles.shape == (4,)
    assert np.all(min_ranges == pytest.approx(SPEC.range_max))
    width = 2 * math.pi / 4
    assert angles == pytest.approx([wrap(width * (i + 0.5)) for i in range(4)])


def test_minimum_and_its_bearing_are_picked_per_sector():
    ranges = np.full(360, 10.0)
    ranges[45] = 1.5  # sector 0 of 4 (0 .. 90 deg)
    ranges[200] = 0.5  # sector 2 (180 .. 270 deg)
    ranges[210] = 0.7
    min_ranges, angles = bin_scan(ranges, SPEC, num_bins=4)
    assert min_ranges == pytest.approx([1.5, 10.0, 0.5, 10.0])
    assert angles[0] == pytest.approx(45 * SPEC.angle_increment)
    assert angles[2] == pytest.approx(wrap(200 * SPEC.angle_increment))


def test_invalid_readings_are_ignored():
    ranges = np.full(360, 5.0)
    ranges[10] = np.nan
    ranges[20] = 0.01  # below range_min
    ranges[30] = 50.0  # above range_max
    ranges[40] = -1.0
    min_ranges, _ = bin_scan(ranges, SPEC, num_bins=1)
    assert min_ranges == pytest.approx([5.0])


def test_num_bins_is_clamped_to_number_of_rays():
    ranges = np.linspace(1.0, 2.0, 10)
    spec = ScanSpec(0.0, 1.0, 0.1, 5.0, num_ranges=10)
    min_ranges, _ = bin_scan(ranges, spec, num_bins=50)
    assert min_ranges.shape == (10,)
    assert min_ranges == pytest.approx(ranges)


def test_matches_reference_loop_implementation():
    rng = np.random.default_rng(0)
    ranges = rng.uniform(0.0, 25.0, size=360)
    ranges[rng.integers(0, 360, 20)] = np.inf
    num_bins = 30

    # Reference per-ray loop.
    bin_width = (SPEC.angle_max - SPEC.angle_min) / num_bins
    ref_ranges = [SPEC.range_max] * num_bins
    ref_angles = [SPEC.angle_min + (i + 0.5) * bin_width for i in range(num_bins)]
    for i, r in enumerate(ranges):
        angle = SPEC.angle_min + i * (SPEC.angle_max - SPEC.angle_min) / (
            SPEC.num_ranges - 1
        )
        if not (SPEC.range_min <= r <= SPEC.range_max) or not math.isfinite(r):
            continue
        idx = int(min((angle - SPEC.angle_min) // bin_width, num_bins - 1))
        if r < ref_ranges[idx]:
            ref_ranges[idx] = r
            ref_angles[idx] = angle

    min_ranges, angles = bin_scan(ranges, SPEC, num_bins)
    assert min_ranges == pytest.approx(ref_ranges, rel=1e-6)
    assert angles == pytest.approx([wrap(a) for a in ref_angles], rel=1e-6)


def test_yaw_offset_rotates_and_wraps_bearings():
    spec = ScanSpec(
        angle_min=0.0,
        angle_max=2 * np.pi,
        range_min=0.1,
        range_max=10.0,
        num_ranges=360,
    )
    ranges = np.full(360, 5.0)
    ranges[0] = 1.0  # straight ahead in the lidar frame
    _, angles = bin_scan(ranges, spec, num_bins=36, yaw_offset=np.pi / 2)
    assert angles[0] == pytest.approx(np.pi / 2)
    assert np.all(angles >= -np.pi) and np.all(angles < np.pi)
