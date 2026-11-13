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

import xml.etree.ElementTree as ET

from drl_nav.sim.description import apply_sensor_profile, sensor_names

URDF = """
<robot name="r">
  <link name="base_link"/>
  <gazebo reference="lidar_link">
    <sensor name="rplidar" type="gpu_lidar">
      <update_rate>62.0</update_rate>
      <lidar><scan><horizontal><samples>640</samples></horizontal></scan></lidar>
    </sensor>
  </gazebo>
  <gazebo reference="cliff_left"><sensor name="cliff_left" type="gpu_lidar"/></gazebo>
  <gazebo reference="ir"><sensor name="ir_intensity_left" type="gpu_lidar"/></gazebo>
  <gazebo reference="cam"><sensor name="rgbd_camera" type="rgbd_camera"/></gazebo>
  <gazebo reference="bumper"><sensor name="bumper_contact" type="contact"/></gazebo>
  <gazebo><plugin filename="x" name="y"/></gazebo>
</robot>
"""


def test_profile_removes_and_retunes():
    out = apply_sensor_profile(
        URDF,
        remove_prefixes=("cliff_", "ir_intensity_"),
        lidar={"name": "rplidar", "update_rate": 10, "samples": 360},
        keep_cameras=False,
    )
    names = sensor_names(out)
    assert set(names) == {"rplidar", "bumper_contact"}
    lidar = [
        s for s in ET.fromstring(out).iter("sensor") if s.get("name") == "rplidar"
    ][0]
    assert lidar.findtext("update_rate") == "10"
    assert lidar.find(".//horizontal/samples").text == "360"
    assert ET.fromstring(out).find(".//plugin") is not None


def test_official_profile_is_identity_on_sensors():
    assert sensor_names(apply_sensor_profile(URDF)) == sensor_names(URDF)
