#!/usr/bin/env python

###############################################################################
# Copyright 2018 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import math

import numpy as np

from parameters import dim


def GetDatapoints(pose_slice, cs_slice):
    assert len(cs_slice) == len(pose_slice)
    assert len(cs_slice) > 0
    out = np.zeros([len(cs_slice), dim["pose"]+dim["chassis"]])
    ref_time = pose_slice[0].header.timestamp_sec
    ref_x = pose_slice[0].pose.position.x
    ref_y = pose_slice[0].pose.position.y
    ref_z = pose_slice[0].pose.position.z
    for i in range(len(cs_slice)):
        pose = pose_slice[i].pose
        chassis = cs_slice[i]
        out[i] = np.array([
            pose.heading - 0.11,  # 0
            pose.orientation.qx,  # 1
            pose.orientation.qy,  # 2
            pose.orientation.qz,  # 3
            pose.orientation.qw,  # 4
            pose.linear_velocity.x,  # 5
            pose.linear_velocity.y,  # 6
            pose.linear_velocity.z,  # 7
            pose.linear_acceleration.x,  # 8
            pose.linear_acceleration.y,  # 9
            pose.linear_acceleration.z,  # 10
            pose.angular_velocity.x,  # 11
            pose.angular_velocity.y,  # 12
            pose.angular_velocity.z,  # 13
            chassis.speed_mps,  # 14
            chassis.throttle_percentage/100.0,  # 15
            chassis.brake_percentage/100.0,  # 16
            chassis.steering_percentage/100.0,  # 17
            chassis.driving_mode,  # 18
            pose.position.x,  # 19
            pose.position.y,  # 20
            pose.position.z,  # 21
        ])
    return out
