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
Launch-time edits of a robot description's Gazebo sensors.

Official descriptions expose no arguments for sensor rates or sets, and their full sensor
suites dominate simulation time on machines without a GPU. ``apply_sensor_profile`` edits the
``<gazebo><sensor>`` blocks of the generated URDF in memory.
"""

from typing import Dict, Optional, Sequence
import xml.etree.ElementTree as ET

CAMERA_TYPES = ("camera", "depth_camera", "rgbd_camera", "thermal_camera")


def apply_sensor_profile(
    urdf_xml: str,
    remove_prefixes: Sequence[str] = (),
    lidar: Optional[Dict[str, object]] = None,
    keep_cameras: bool = True,
) -> str:
    """
    Return ``urdf_xml`` with sensors removed or retuned.

    ``remove_prefixes``: sensors whose name starts with one of these are dropped.
    ``lidar``: ``{'name': ..., 'update_rate': ..., 'samples': ...}`` retunes one lidar.
    ``keep_cameras``: when false, every camera-type sensor is dropped.
    """
    root = ET.fromstring(urdf_xml)
    for gazebo in root.iter("gazebo"):
        for sensor in list(gazebo.findall("sensor")):
            name, kind = sensor.get("name", ""), sensor.get("type", "")
            drop_camera = not keep_cameras and kind in CAMERA_TYPES
            if name.startswith(tuple(remove_prefixes)) or drop_camera:
                gazebo.remove(sensor)
            elif lidar and name == lidar.get("name"):
                _set_text(sensor, "update_rate", lidar.get("update_rate"))
                _set_text(sensor, ".//horizontal/samples", lidar.get("samples"))
    return ET.tostring(root, encoding="unicode")


def sensor_names(urdf_xml: str) -> Dict[str, str]:
    """Map sensor name -> type for the ``<gazebo><sensor>`` blocks of a URDF."""
    root = ET.fromstring(urdf_xml)
    return {
        s.get("name", ""): s.get("type", "")
        for g in root.iter("gazebo")
        for s in g.findall("sensor")
    }


def _set_text(sensor: ET.Element, path: str, value) -> None:
    if value is None:
        return
    element = sensor.find(path)
    if element is not None:
        element.text = str(value)
