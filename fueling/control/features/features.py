#!/usr/bin/env python

import math
import sys

import numpy as np

from fueling.control.features.parameters import dim
import fueling.common.colored_glog as glog

def GetDatapoints(pose_slice, cs_slice):
    if len(cs_slice) == 0:
        glog.info('Feature Extraction Error: Control Slice Empty')
        sys.exit(1)
    out = np.zeros([len(cs_slice), dim["pose"] + dim["chassis"]])
    ref_time = pose_slice[0].header.timestamp_sec
    ref_x = pose_slice[0].pose.position.x
    ref_y = pose_slice[0].pose.position.y
    ref_z = pose_slice[0].pose.position.z
    for i in range(len(cs_slice)):
        pose = pose_slice[i].pose
        chassis = cs_slice[i]
        out[i] = np.array([
            # TODO: need to load IMU compensation from file
            pose.heading,  # 0
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
