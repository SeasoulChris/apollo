#!/usr/bin/env python


import numpy as np

import fueling.common.proto_utils as proto_utils
import fueling.common.logging as logging
from modules.localization.proto import localization_pb2


SEGMENT_LEN = 2  # eight sec
SEGMENT_INTERVAL = 1  # shift one frame


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
    """ extract localization from msg """
    return (localization_msg.header, gen_data_point(localization_msg.pose))


def partition_data(target_msgs, segment_len=SEGMENT_LEN, segment_int=SEGMENT_INTERVAL):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    logging.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msgs: msgs.timestamp)
    msgs_groups = [msgs[idx: idx + segment_len]
                   for idx in range(0, len(msgs), segment_int)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]
