#!/usr/bin/env python

""" Open-space planner feature extraction related utils. """


import numpy as np
import math

import fueling.common.logging as logging
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils
from fueling.profiling.conf.open_space_planner_conf import FEATURE_IDX, REFERENCE_VALUES

def delta(feature_name, prev_feature, curr_feature):
    feature_idx = FEATURE_IDX[feature_name]
    return curr_feature[feature_idx] - prev_feature[feature_idx]


def calc_lon_acc_bound(acc_lat, dec_lat):
    if acc_lat == 0.0:
        return 0.5 * (dec_lat + 3.0)
    return -0.5 * (acc_lat - 3.0)


def calc_lon_dec_bound(acc_lat, dec_lat):
    if acc_lat == 0.0:
        return -2.0 / 3.0 * (dec_lat + 3.0)
    return 2.0 / 3.0 * (acc_lat - 3.0)


def calc_lat_acc_bound(acc_lon, dec_lon):
    if acc_lon == 0.0:
        return 3.0 / 2.0 * dec_lon + 3.0
    return -2.0 * acc_lon + 3.0


def calc_lat_dec_bound(acc_lon, dec_lon):
    if acc_lon == 0.0:
        return -3.0 / 2.0 * dec_lon - 3.0
    return 2 * acc_lon - 3.0


def steer_limit(steer_val, vehicle_param):
    return steer_val / vehicle_param.steer_ratio / vehicle_param.wheel_base


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
    lat = speed * speed * path_point.kappa
    if lat > 0.0:
        lat_acc = lat
        lat_dec = 0.0
    else:
        lat_acc = 0.0
        lat_dec = lat
    # get longitudinal acc and dec
    if a > 0.0:
        lon_acc = math.sqrt(abs(a**2 - (lat)**2))
        lon_dec = 0.0
    else:
        lon_acc = 0.0
        lon_dec = -1.0 * math.sqrt(abs(a**2 - (lat)**2))

    # calculate comfort bound
    lon_acc_bound = calc_lon_acc_bound(lat_acc, lat_dec)
    lon_dec_bound = calc_lon_dec_bound(lat_acc, lat_dec)
    lat_acc_bound = calc_lat_acc_bound(lon_acc, lon_dec)
    lat_dec_bound = calc_lat_dec_bound(lon_acc, lon_dec)

    if hasattr(trajectory_point, 'relative_time'):
        # NOTE: make sure to update TRAJECTORY_FEATURE_NAMES in
        # open_space_planner_conf.py if updating this data array.
        # Will need a better way to sync these two pieces.
        data_array = np.array([
            trajectory_point.relative_time,
            path_point.kappa,
            abs(path_point.kappa) / steer_limit(vehicle_param.max_steer_angle, vehicle_param),
            speed,  # not sure if needed
            a,
            lon_acc if a > 0.0 else lon_dec,
            lat_acc if lat > 0.0 else lat_dec,

            # ratios
            a / vehicle_param.max_acceleration if a > 0.0 else 0.0,
            a / vehicle_param.max_deceleration if a < 0.0 else 0.0,
            lon_acc / lon_acc_bound,
            lon_dec / lon_dec_bound,
            lat_acc / lat_acc_bound,
            lat_dec / lat_dec_bound,
        ])
    return data_array


def calculate_jerk_ratios(prev_feature, curr_feature):
    if prev_feature is None or curr_feature is None:
        return [0.0, 0.0, 0.0, 0.0]


    delta_t = delta('relative_time', prev_feature, curr_feature)
    lon_jerk = delta('longitudinal_acceleration', prev_feature, curr_feature) / delta_t
    if lon_jerk > 0.0:
        lon_pos_jerk = lon_jerk
        lon_neg_jerk = 0.0
    else:
        lon_pos_jerk = 0.0
        lon_neg_jerk = lon_jerk

    lat_jerk = delta('lateral_acceleration', prev_feature, curr_feature) / delta_t
    if lat_jerk > 0.0:
        lat_pos_jerk = lat_jerk
        lat_neg_jerk = 0.0
    else:
        lat_pos_jerk = 0.0
        lat_neg_jerk = lat_jerk

    return [
        lon_pos_jerk / REFERENCE_VALUES['longitudinal_jerk_positive_upper_bound'],
        lon_neg_jerk / REFERENCE_VALUES['longitudinal_jerk_negative_upper_bound'],
        lat_pos_jerk / REFERENCE_VALUES['lateral_jerk_positive_upper_bound'],
        lat_neg_jerk / REFERENCE_VALUES['lateral_jerk_negative_upper_bound'],
    ]


def calculate_dkappa_ratio(prev_feature, curr_feature, vehicle_param):
    if prev_feature is None or curr_feature is None:
        return 0.0

    delta_t = delta('relative_time', prev_feature, curr_feature)
    dkappa = delta('kappa', prev_feature, curr_feature) / delta_t
    dkappa_ratio = abs(dkappa) / steer_limit(vehicle_param.max_steer_angle_rate, vehicle_param),
    return [dkappa_ratio]


def extract_data_from_trajectory(trajectory, vehicle_param):
    """Extract data from all trajectory points"""
    feature_list = []
    prev_features = None
    for trajectory_point in trajectory:
        features = extract_data_from_trajectory_point(trajectory_point, vehicle_param)
        if features is None:
            continue

        features = np.append(features, calculate_jerk_ratios(prev_features, features))
        features = np.append(features, calculate_dkappa_ratio(prev_features, features, vehicle_param))
        feature_list.append(features)
        prev_features = features

    trajectory_mtx = np.array(feature_list)
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
    zigzag_latency = 0.0
    if msg.debug.planning_data.open_space.time_latency:
        zigzag_latency = msg.debug.planning_data.open_space.time_latency
    meta_array = np.array([
        msg.latency_stats.total_time_ms,  # end-to-end time latency
        zigzag_latency,  # zigzag trajectory time latency
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
    return target, group_id, np.array([zigzag_list]).T  # make sure numpy shape is (num, 1)


def extract_stage_feature(target_groups):
    """Extract scenario stage related feature matrix from a group of planning messages"""
    target, group_id, msgs = target_groups
    logging.info(F'Computing {len(msgs)} messages for target {target}')
    
    start_timestamp = msgs[0].header.timestamp_sec
    end_timestamp = msgs[-1].header.timestamp_sec
    gear_shift_times = 1
    for msg in msgs:
        # Find the first frame when zigzag trajectory is ready
        zigzag_count = len(msg.debug.planning_data.open_space.partitioned_trajectories.trajectory)
        if zigzag_count > 0:
            gear_shift_times = zigzag_count
            for point in msg.trajectory_point:
                if point.relative_time == 0.0:
                    initial_heading = point.path_point.theta
                    break
            break
    stage_completion_time = (end_timestamp - start_timestamp) / gear_shift_times * 1000.0

    actual_heading = msgs[0].debug.planning_data.adc_position.pose.heading
    vehicle_param = multi_vehicle_utils.get_vehicle_param(target)
    initial_heading_diff_ratio = abs(initial_heading - actual_heading) \
                                 / (vehicle_param.max_steer_angle / vehicle_param.steer_ratio)
    return target, group_id, np.array([[stage_completion_time, initial_heading_diff_ratio]])
