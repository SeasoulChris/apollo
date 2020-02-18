#!/usr/bin/env python


import numpy as np

import fueling.common.proto_utils as proto_utils
from modules.localization.proto import localization_pb2


def gen_data_point(pose):
    return np.array([pose.heading,  # 0
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
                     pose.position.x,  # 14
                     pose.position.y,  # 15
                     pose.position.z,  # 16
                     ])


def LoadEgoCarLocalization(localization_msg):
    """ extract trajectory from a record file """
    return (localization_msg.header, gen_data_point(localization_msg.pose))
