#!/usr/bin/env python

""" Open-space planner feature extraction related utils. """


import numpy as np
import math

import fueling.common.logging as logging
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils


# trajectory point example:
# path_point:
#     x: 559787.066030095
#     y: 4157751.813925536
#     z: 0.000000000
#     theta: 2.379002832
#     kappa: -0.019549716
#     s: -2.356402468
#     dkappa: 0.000000000
#     ddkappa: 0.000000000
# v: 4.474370468
# a: 0.995744297
# relative_time: -0.400000000
def extract_data_from_trajectory_point(trajectory_point, vehicle_param):
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
            speed,  # not sure if needed
            a,
            a / vehicle_param.max_acceleration if a > 0.0 else 0.0,
            a / vehicle_param.max_deceleration if a < 0.0 else 0.0,
            lat_acc / lat_acc_bound,
        ])
    return data_array


def extract_data_from_trajectory(trajectory, vehicle_param):
    """Extract data from all trajectory points"""
    extract_list = (extract_data_from_trajectory_point(trajectory_point, vehicle_param)
                    for trajectory_point in trajectory)
    trajectory_mtx = np.array([data for data in extract_list if data is not None])
    return trajectory_mtx


def extract_planning_trajectory_feature(target_groups):
    """Extract planning trajectory related feature matrix from a group of planning messages"""
    target, group_id, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')

    vehicle_param = multi_vehicle_utils.get_vehicle_param(target)

    extracted_data = (extract_data_from_trajectory(msg.trajectory_point, vehicle_param)
                      for msg in msgs)
    planning_trajectory_mtx = np.concatenate(
        [data for data in extracted_data if data is not None and data.shape[0] > 10])

    return target, group_id, planning_trajectory_mtx


def extract_meta_from_planning(msg):
    """Extract non-repeated field from one planning message"""
    meta_array = np.array([
        msg.latency_stats.total_time_ms,  # end-to-end time latency
        msg.debug.planning_data.open_space.time_latency,  # zigzag trajectory latency
        # TODO stage completed time: equation needs confirmation
    ])
    return meta_array


def extract_latency_feature(target_groups):
    """Extract latency related feature matrix from a group of planning messages"""
    target, group_id, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')
    latency_mtx = np.array([data for data in [extract_meta_from_planning(msg)
                                              for msg in msgs] if data is not None])
    return target, group_id, latency_mtx


def compute_path_length(trajectory):
    trajectory_points = trajectory.trajectory_point
    if len(trajectory_points) < 2:
        return 0.0
    return abs(trajectory_points[-1].path_point.s - trajectory_points[0].path_point.s)


def extract_data_from_zigzag(msg, wheel_base):
    """Extract open space debug from one planning message"""
    data = [compute_path_length(zigzag) / wheel_base
            for zigzag in msg.debug.planning_data.open_space.partitioned_trajectories.trajectory]
    return data


def extract_zigzag_trajectory_feature(target_groups):
    """Extract zigzag trajectory related feature matrix from a group of planning messages"""
    target, group_id, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')

    vehicle_param = multi_vehicle_utils.get_vehicle_param(target)

    zigzag_list = []
    for msg in msgs:
        zigzag_list.extend(extract_data_from_zigzag(msg, vehicle_param.wheel_base))
    return target, group_id, np.array([zigzag_list]).T # make sure numpy shape is (num, 1)


def calc_lat_acc_bound(acc_lon, dec_lon):
    if acc_lon == 0.0:
        return 3.0 / 2.0 * dec_lon + 3.0
    return -2.0 * acc_lon + 3.0
