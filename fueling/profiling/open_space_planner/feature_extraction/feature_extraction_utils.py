#!/usr/bin/env python

""" Open-space planner feature extraction related utils. """


import numpy as np

import fueling.common.logging as logging

from modules.planning.proto.planning_config_pb2 import ScenarioConfig


def extract_planning_data_from_msg(msg):
    """Extract fields from planning message"""
    data_array = np.array([
        # Features: "Header" category
        msg.header.timestamp_sec,
        msg.header.sequence_num,
        # Features: "Latency" category
        msg.latency_stats.total_time_ms,
    ])
    return data_array


def extract_mtx(target_groups):
    target, group_id, msgs = target_groups
    logging.info('computing {} messages for target {}'.format(len(msgs), target))
    planning_mtx = np.array([data for data in [extract_planning_data_from_msg(msg)
                                               for msg in msgs] if data is not None])
    return planning_mtx
