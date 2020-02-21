#!/usr/bin/env python

""" Open-space planner feature extraction related utils. """


import numpy as np
import math

from fueling.profiling.proto.open_space_planner_profiling_pb2 import OpenSpacePlannerProfiling
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


def get_config_open_space_profiling():
    """Get configured value in open_space_profiling_conf.pb.txt"""
    profiling_conf = '/fuel/fueling/profiling/conf/open_space_planner_profiling_conf.pb.txt'
    open_space_planner_profiling = OpenSpacePlannerProfiling()
    proto_utils.get_pb_from_text_file(profiling_conf, open_space_planner_profiling)
    return open_space_planner_profiling

#  trajectory points
#  path_point:
#     x: 559787.066030095
#     y: 4157751.813925536
#     z: 0.000000000
#     theta: 2.379002832
#     kappa: -0.019549716
#     s: -2.356402468
#     dkappa: 0.000000000
#     ddkappa: 0.000000000
#   v: 4.474370468
#   a: 0.995744297
#   relative_time: -0.400000000


def extract_data_from_trajectory_point(trajectory_point):
    """Extract fields from a single trajectory point"""
    path_point = trajectory_point.path_point
    speed = trajectory_point.v
    a = trajectory_point.a
    # get lateral acc and dec
    if speed * speed * path_point.kappa > 0.0:
        lat_acc = speed * speed * path_point.kappa
        lat_dec = 0.0
    else:
        lat_acc = 0.0
        lat_dec = speed * speed * path_point.kappa
    # get longitudinal acc and dec
    if a > 0.0:
        lon_acc = math.sqrt(abs(a**2 - (speed * speed * path_point.kappa)**2))
        lon_dec = 0.0
    else:
        lon_acc = 0.0
        lon_dec = -1.0 * math.sqrt(abs(a**2 - (speed * speed * path_point.kappa)**2))
    # calculate lateral acceleration bound
    lat_acc_bound = calc_lat_acc_bound(lon_acc, lon_dec)
    if hasattr(trajectory_point, 'relative_time'):
        data_array = np.array([
            trajectory_point.relative_time,
            speed,
            trajectory_point.a,
            lat_acc,
            lat_dec,
            lon_acc,
            lon_dec,
            lat_acc >= lat_acc_bound,
        ])
    return data_array


def extract_data_from_all_trajectory_point(msg):
    """Extract data from all trajectory points"""
    trajectory_points = msg.trajectory_point
    # logging.info('computing {} trajectory_point from frame No {}'.format(
    #     len(trajectory_points), msg.header.sequence_num))
    trajectory_mtx = np.array([data for data in [extract_data_from_trajectory_point(trajectory_point)
                                                 for trajectory_point in trajectory_points] if data is not None])
    return trajectory_mtx


def extract_planning_data_from_msg(msg):
    """Extract non-repeated field from planning message"""
    data_array = np.array([
        # Features: "Header" category
        msg.header.timestamp_sec,
        msg.header.sequence_num,
        # Features: "Latency" category
        msg.latency_stats.total_time_ms,
    ])
    return data_array


def extract_mtx(target_groups):
    """Extract matrix data of non-repeated fields from a group of messages"""
    target, group_id, msgs = target_groups
    logging.info('computing {} messages for target {}'.format(len(msgs), target))
    planning_mtx = np.array([data for data in [extract_planning_data_from_msg(msg)
                                               for msg in msgs] if data is not None])
    return target, group_id, planning_mtx


def extract_mtx_repeated_field(target_groups):
    """Extract matrix data of repeated fields from a group of messages"""
    target, group_id, msgs = target_groups
    logging.info('computing {} messages for target {}'.format(len(msgs), target))
    planning_mtx = np.array([data for data in (extract_data_from_all_trajectory_point(msg)
                                               for msg in msgs) if data is not None])
    planning_mtx = extract_data_from_all_trajectory_point(msgs[0])
    for msg in msgs[1:-1]:
        data = extract_data_from_all_trajectory_point(msg)
        # 10 trajectory point is stop trajectory
        if(data.shape[0] > 10):
            planning_mtx = np.concatenate((planning_mtx, data), axis=0)
    return target, group_id, planning_mtx


def calc_lat_acc_bound(acc_lon, dec_lon):
    if acc_lon == 0.0:
        return 3.0 / 2.0 * dec_lon + 3.0
    else:
        return -2.0 * acc_lon + 3.0
