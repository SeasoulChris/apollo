#!/usr/bin/env python

""" Open-space planner feature extraction related utils. """


import numpy as np


from modules.data.fuel.fueling.profiling.proto.open_space_planner_profiling_pb2 import OpenSpacePlannerProfiling
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

from modules.planning.proto.planning_config_pb2 import ScenarioConfig


def get_config_open_space_profiling():
    """Get configured value in open_space_profiling_conf.pb.txt"""
    profiling_conf = \
        '/apollo/modules/data/fuel/fueling/profiling/conf/open_space_planner_profiling_conf.pb.txt'
    open_space_planner_profiling = OpenSpacePlannerProfiling()
    proto_utils.get_pb_from_text_file(profiling_conf, open_space_planner_profiling)
    return open_space_planner_profiling


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
    return target, group_id, planning_mtx
