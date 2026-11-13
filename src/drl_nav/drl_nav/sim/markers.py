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

"""Builders for the RViz markers used to visualise observations."""

from typing import Dict, Sequence, Tuple

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose
import numpy as np
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def goal_marker(
    goal_pose: Pose, frame_id: str, stamp: Time, marker_id: int = 0
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "goal"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose = goal_pose
    marker.scale.x = 0.5
    marker.scale.y = 0.05
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.b = 1.0
    return marker


def observation_markers(
    observation: Dict[str, np.ndarray],
    robot_xy_yaw: Tuple[float, float, float],
    goal_xy: Tuple[float, float],
    world_frame: str,
    base_frame: str,
    stamp: Time,
    robot_radius: float = 1.0,
) -> MarkerArray:
    """Draw the binned lidar rays, the goal direction and the heading error."""
    robot_x, robot_y, robot_yaw = robot_xy_yaw

    world_header = Header(frame_id=world_frame, stamp=stamp)
    base_header = Header(frame_id=base_frame, stamp=stamp)

    cos_y, sin_y = np.cos(robot_yaw), np.sin(robot_yaw)
    transform = np.array(
        [
            [cos_y, -sin_y, robot_x],
            [sin_y, cos_y, robot_y],
            [0.0, 0.0, 1.0],
        ]
    )

    markers = MarkerArray()
    robot_point = Point(x=robot_x, y=robot_y, z=0.0)

    sectors = zip(observation["min_ranges"], observation["min_ranges_angle"])
    for i, (rng, ang) in enumerate(sectors):
        local = np.array([rng * np.cos(ang), rng * np.sin(ang), 1.0])
        world = transform @ local
        marker = Marker()
        marker.header = world_header
        marker.ns = "min_ranges"
        marker.id = i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.points = [
            robot_point,
            Point(x=float(world[0]), y=float(world[1]), z=0.0),
        ]
        markers.markers.append(marker)

    markers.markers.append(
        _arrow(
            robot_point,
            Point(x=goal_xy[0], y=goal_xy[1], z=0.0),
            world_header,
            ns="orient_arrow",
            marker_id=0,
            rgb=(0.0, 1.0, 0.0),
        )
    )
    ahead = transform @ np.array([2.0, 0.0, 1.0])
    markers.markers.append(
        _arrow(
            robot_point,
            Point(x=float(ahead[0]), y=float(ahead[1]), z=0.0),
            world_header,
            ns="orient_arrow",
            marker_id=1,
            rgb=(0.0, 1.0, 0.0),
        )
    )

    orient_to_goal = float(observation["orient_to_goal"][0])
    markers.markers.append(
        _arc(radius=robot_radius, theta_end=orient_to_goal, header=base_header)
    )
    markers.markers.append(
        _text(
            text=f"{orient_to_goal:.3f} rad",
            theta=orient_to_goal / 2,
            radius=robot_radius + 0.5,
            header=base_header,
        )
    )
    return markers


def _arrow(
    start: Point,
    end: Point,
    header: Header,
    ns: str,
    marker_id: int,
    rgb: Sequence[float],
) -> Marker:
    marker = Marker()
    marker.header = header
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.id = marker_id
    marker.ns = ns
    marker.scale.x = 0.05
    marker.scale.y = 0.1
    marker.scale.z = 0.2
    marker.color.r, marker.color.g, marker.color.b = rgb
    marker.color.a = 1.0
    marker.points = [start, end]
    return marker


def _arc(
    radius: float, theta_end: float, header: Header, num_points: int = 50
) -> Marker:
    marker = Marker()
    marker.header = header
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.ns = "arc"
    marker.id = 0
    marker.scale.x = 0.02
    marker.color.g = 1.0
    marker.color.a = 1.0
    for theta in np.linspace(0.0, theta_end, num_points + 1):
        marker.points.append(
            Point(
                x=float(radius * np.cos(theta)), y=float(radius * np.sin(theta)), z=0.0
            )
        )
    return marker


def _text(text: str, theta: float, radius: float, header: Header) -> Marker:
    marker = Marker()
    marker.header = header
    marker.type = Marker.TEXT_VIEW_FACING
    marker.ns = "text"
    marker.id = 0
    marker.text = text
    marker.scale.z = 0.25
    marker.color.r = marker.color.g = marker.color.b = 1.0
    marker.color.a = 1.0
    marker.pose.position.x = float(radius * np.cos(theta))
    marker.pose.position.y = float(radius * np.sin(theta))
    marker.pose.position.z = 0.1
    return marker
