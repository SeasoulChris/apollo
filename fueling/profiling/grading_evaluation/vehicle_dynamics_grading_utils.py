#!/usr/bin/env python
""" Control performance grading related utils. """

from collections import namedtuple
import math
import numpy as np
import os

from fueling.profiling.conf.control_channel_conf import DYNAMICS_FEATURE_IDX
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.profiling.feature_extraction.vehicle_dynamics_feature_extraction_utils as feature_utils


def generating_matrix_and_h5(target_groups):
    """Do computing against one group"""
    target, group_id, msgs = target_groups
    logging.info('computing {} messages for target {}'.format(len(msgs), target))
    profiling_conf = feature_utils.get_profiling_config()
    grading_mtx = feature_utils.extract_data_two_channels(msgs, profiling_conf.driving_mode,
                                                          profiling_conf.gear_position)
    if grading_mtx.shape[0] == 0:
        logging.warning('no valid element in {} items in group {} for task {}'
                        .format(len(msgs), group_id, target))
        return (target, None)
    h5_output_file = '{}_{}_{:05d}'.format(profiling_conf.vehicle_type,
                                           profiling_conf.controller_type,
                                           group_id)
    logging.info('writing {} messages to h5 file {} for target {}'
                 .format(grading_mtx.shape[0], h5_output_file, target))
    h5_utils.write_h5(grading_mtx, target, h5_output_file)
    logging.info('grading_mtx information: {}'
                 .format(grading_mtx[0]))
    return (grading_mtx, target)


def computing_gradings(mtx_groups):
    """Do computing against one group"""
    grading_mtx, target = mtx_groups
    grading_results = namedtuple('grading_results',
                                 ['throttle_dead_time',
                                  'throttle_rise_time',
                                  'throttle_setting_time',
                                  'throttle_overshoot',
                                  'throttle_bandwidth',
                                  'throttle_resonant_peak',
                                  'brake_dead_time',
                                  'brake_rise_time',
                                  'brake_setting_time',
                                  'brake_overshoot',
                                  'brake_bandwidth',
                                  'brake_resonant_peak',
                                  'steering_dead_time',
                                  'steering_rise_time',
                                  'steering_setting_time',
                                  'steering_overshoot',
                                  'steering_bandwidth',
                                  'steering_resonant_peak',
                                  ])
    grading_arguments = namedtuple('grading_arguments',
                                   ['std_filter_name',
                                    'std_filter_value',
                                    'std_filter_mode',
                                    'std_norm_name',
                                    'std_denorm_name',
                                    'std_max_compare',
                                    'std_denorm_weight',
                                    'peak_feature_name',
                                    'peak_time_name',
                                    'peak_threshold',
                                    'peak_filter_name',
                                    'peak_filter_value',
                                    'peak_filter_mode',
                                    'ending_feature_name',
                                    'ending_time_name',
                                    'ending_filter_name',
                                    'ending_filter_value',
                                    'ending_filter_mode',
                                    'ending_threshold',
                                    'usage_feature_name',
                                    'usage_filter_name',
                                    'usage_filter_value',
                                    'usage_filter_mode',
                                    'usage_weight',
                                    'mean_feature_name',
                                    'mean_filter_name',
                                    'mean_filter_value',
                                    'mean_filter_mode',
                                    'mean_weight',
                                    'beyond_feature_name',
                                    'beyond_threshold',
                                    'count_feature_name',
                                    'desired_feature_name',
                                    'measured_feature_name'])

    grading_results.__new__.__defaults__ = (
        None,) * len(grading_results._fields)
    grading_arguments.__new__.__defaults__ = (
        None,) * len(grading_arguments._fields)

    # Compute time domain fields
    throttle_rise_time, throttle_overshoot, throttle_setting_time = compute_dynamics_time(
        grading_mtx, grading_arguments(
            desired_feature_name='acceleration_cmd',
            measured_feature_name='acceleration'
        ))
    brake_rise_time, brake_overshoot, brake_setting_time = compute_dynamics_time(
        grading_mtx, grading_arguments(
            desired_feature_name='deceleration_cmd',
            measured_feature_name='deceleration'
        ))
    steering_rise_time, steering_overshoot, steering_setting_time = compute_dynamics_time(
        grading_mtx, grading_arguments(
            desired_feature_name='streeting_cmd',
            measured_feature_name='streering'
        ))
    # Compute frequency domain field
    throttle_bandwidth, throttle_resonant_peak = compute_dynamics_freq(
        grading_mtx, grading_arguments(
            desired_feature_name='acceleration_cmd',
            measured_feature_name='acceleration'
        ))
    brake_bandwidth, brake_resonant_peak = compute_dynamics_freq(
        grading_mtx, grading_arguments(
            desired_feature_name='deceleration_cmd',
            measured_feature_name='deceleration'
        ))
    steering_bandwidth, steering_resonant_peak = compute_dynamics_freq(
        grading_mtx, grading_arguments(
            desired_feature_name='streeting_cmd',
            measured_feature_name='streering'
        ))
    grading_group_result = grading_results(
        throttle_dead_time=compute_deadtime(),
        throttle_rise_time=throttle_rise_time,
        throttle_overshoot=throttle_overshoot,
        throttle_setting_time=throttle_setting_time,
        throttle_bandwidth=throttle_bandwidth,
        throttle_resonant_peak=throttle_resonant_peak,
        brake_dead_time=compute_deadtime(),
        brake_rise_time=brake_rise_time,
        brake_overshoot=brake_overshoot,
        brake_setting_time=brake_setting_time,
        brake_bandwidth=brake_bandwidth,
        brake_resonant_peak=brake_resonant_peak,
        steering_dead_time=compute_deadtime(),
        steering_rise_time=steering_rise_time,
        steering_overshoot=steering_overshoot,
        steering_setting_time=steering_setting_time,
        steering_bandwidth=steering_bandwidth,
        steering_resonant_peak=steering_resonant_peak,
    )
    return (target, grading_group_result)


def compute_deadtime():
    # TODO(fengzongbao): Deadtime need to be calculated
    deadtime = 0.0
    return deadtime


def compute_dynamics_time(grading_mtx, arg):
    """Compute the dynamics time domain field"""

    # measured value
    y = grading_mtx[:, DYNAMICS_FEATURE_IDX[arg.std_norm_name]]
    # desired value
    u = grading_mtx[:, DYNAMICS_FEATURE_IDX[arg.std_denorm_name]]
    # TODO(fengzongbao): Need to complete
    t_dead = 0.0
    a1, a2, b1, b2 = dynamics_estimator(y, u, t_dead)
    natural_freq, damping_ratio = dynamics_transformation(a1, a2, b1, b2)

    rise_time = 1.8 / natural_freq
    overshoot = np.exp(-np.pi * damping_ratio/(1-damping_ratio**2))
    settling_time = 4.6 / (natural_freq * damping_ratio)
    return (rise_time, overshoot, settling_time)


def compute_dynamics_freq(grading_mtx, arg):
    """Compute the dynamics frequency domain field"""

    # measured value
    y = grading_mtx[:, DYNAMICS_FEATURE_IDX[arg.std_norm_name]]
    # desired value
    u = grading_mtx[:, DYNAMICS_FEATURE_IDX[arg.std_denorm_name]]

    # TODO(fengzongbao): Need to complete
    t_dead = 0.0
    a1, a2, b1, b2 = dynamics_estimator(y, u, t_dead)
    natural_freq, damping_ratio = dynamics_transformation(a1, a2, b1, b2)

    bandwidth = natural_freq
    resonant_peak = 0.5 * damping_ratio

    return (bandwidth, resonant_peak)


def dynamics_transformation(a1, a2, b1, b2):
    # TODO(fengzongbao): Need to complete
    natural_freq = 0.0
    damping_ratio = 0.0
    return (natural_freq, damping_ratio)


def dynamics_estimator(y, u, t_dead):
    """sovle the equation"""
    # TODO(fengzongbao): Need to complete
    a1 = 0
    a2 = 0
    b1 = 0
    b2 = 0
    return (a1, a2, b1, b2)


def combine_gradings(grading_x, grading_y):
    """Reduce gradings by combining the groups with different strategies"""
    if not grading_x:
        return grading_y
    elif not grading_y:
        return grading_x
    grading_group_value = []
    for idx in range(len(grading_x._fields)):
        val_x, num_x = grading_x[idx]
        val_y, num_y = grading_y[idx]
        # Standard deviation and usage values
        if num_x + num_y != 0:
            grading_item_value = ((val_x ** 2 * (num_x - 1) + val_y ** 2 * (num_y - 1))
                                  / (num_x + num_y - 1)) ** 0.5

            grading_item_value = (
                val_x * num_x + val_y * num_y) / (num_x + num_y)
        else:
            grading_item_value = 0.0

        # Peak values
        if val_x[0] >= val_y[0]:
            grading_item_value = val_x
        else:
            grading_item_value = val_y

        grading_item_num = num_x + num_y
        grading_group_value.append((grading_item_value, grading_item_num))
    grading_x = grading_x._make(grading_group_value)
    return grading_x


def output_gradings(target_grading):
    """Write the grading results to files in coresponding target dirs"""
    # TODO(fengzongbao): Need to complete
    pass
