#!/usr/bin/env python

""" Control performance grading related utils. """
import math
import numpy as np
import os

from collections import namedtuple

import colored_glog as glog
import fueling.common.h5_utils as h5_utils

from fueling.control.control_profiling.conf.control_channel_conf import FEATURE_IDX
import fueling.control.control_profiling.feature_extraction.control_feature_extraction_utils \
       as feature_utils


def compute_h5_and_gradings(target_groups):
    """Do computing against one group"""
    target, group_id, msgs = target_groups
    glog.info('computing {} messages for target {}'.format(len(msgs), target))
    profiling_conf = feature_utils.get_config_control_profiling()
    grading_mtx = feature_utils.extract_data_at_multi_channels(msgs, profiling_conf.driving_mode,
                                                               profiling_conf.gear_position)
    if grading_mtx.shape[0] == 0:
        glog.warn('no valid element in {} items in group {} for task {}'
                  .format(len(msgs), group_id, target))
        return (target, None)
    h5_output_file = '{}_{}_{:05d}'.format(profiling_conf.vehicle_type,
                                           profiling_conf.controller_type,
                                           group_id)
    glog.info('writing {} messages to h5 file {} for target {}'
              .format(grading_mtx.shape[0], h5_output_file, target))
    h5_utils.write_h5(grading_mtx, target, h5_output_file)
    grading_results = namedtuple('grading_results',
                                 ['station_err_std',
                                  'station_err_std_harsh',
                                  'speed_err_std',
                                  'speed_err_std_harsh',
                                  'lateral_err_std',
                                  'lateral_err_std_harsh',
                                  'lateral_err_rate_std',
                                  'lateral_err_rate_std_harsh',
                                  'heading_err_std',
                                  'heading_err_std_harsh',
                                  'heading_err_rate_std',
                                  'heading_err_rate_std_harsh',
                                  'station_err_peak',
                                  'speed_err_peak',
                                  'lateral_err_peak',
                                  'lateral_err_rate_peak',
                                  'heading_err_peak',
                                  'heading_err_rate_peak',
                                  'ending_station_err',
                                  'ending_lateral_err',
                                  'ending_heading_err',
                                  'acc_bad_sensation',
                                  'jerk_bad_sensation',
                                  'lateral_acc_bad_sensation',
                                  'lateral_jerk_bad_sensation',
                                  'heading_acc_bad_sensation',
                                  'heading_jerk_bad_sensation',
                                  'throttle_control_usage',
                                  'throttle_control_usage_harsh',
                                  'brake_control_usage',
                                  'brake_control_usage_harsh',
                                  'steering_control_usage',
                                  'steering_control_usage_harsh',
                                  'throttle_deadzone_mean',
                                  'brake_deadzone_mean',
                                  'total_time_usage',
                                  'total_time_peak',
                                  'total_time_exceeded_count',
                                  'pose_heading_offset_std',
                                  'pose_heading_offset_peak'])
    grading_arguments = namedtuple('grading_arguments',
                                   ['std_filter_name',
                                    'std_filter_value',
                                    'std_norm_name',
                                    'std_denorm_name',
                                    'std_max_compare',
                                    'std_denorm_weight',
                                    'peak_feature_name',
                                    'peak_threshold',
                                    'peak_filter_name',
                                    'peak_filter_value',
                                    'ending_feature_name',
                                    'ending_time_name',
                                    'ending_filter_name',
                                    'ending_filter_value',
                                    'ending_threshold',
                                    'usage_feature_name',
                                    'usage_thold_value',
                                    'usage_threshold',
                                    'usage_weight',
                                    'mean_feature_name',
                                    'mean_filter_name',
                                    'mean_filter_value',
                                    'mean_weight',
                                    'beyond_feature_name',
                                    'beyond_threshold',
                                    'count_feature_name'])
    grading_results.__new__.__defaults__ = (None,) * len(grading_results._fields)
    grading_arguments.__new__.__defaults__ = (None,) * len(grading_arguments._fields)
    grading_group_result = grading_results(
        station_err_std=compute_std(grading_mtx, grading_arguments(
            std_filter_name='speed_reference',
            std_filter_value=profiling_conf.control_metrics.speed_stop,
            std_norm_name='station_error',
            std_denorm_name=['speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        )),
        station_err_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='acceleration_reference',
            std_filter_value=profiling_conf.control_metrics.acceleration_harsh_limit,
            std_norm_name='station_error',
            std_denorm_name=['speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        )),
        speed_err_std=compute_std(grading_mtx, grading_arguments(
            std_filter_name='speed_reference',
            std_filter_value=profiling_conf.control_metrics.speed_stop,
            std_norm_name='speed_error',
            std_denorm_name=['speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.speed_still],
            std_denorm_weight=1.0
        )),
        speed_err_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='acceleration_reference',
            std_filter_value=profiling_conf.control_metrics.acceleration_harsh_limit,
            std_norm_name='speed_error',
            std_denorm_name=['speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.speed_still],
            std_denorm_weight=1.0
        )),
        lateral_err_std=compute_std(grading_mtx, grading_arguments(
            std_filter_name='speed_reference',
            std_filter_value=profiling_conf.control_metrics.speed_stop,
            std_norm_name='lateral_error',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
                              * profiling_conf.vehicle_wheelbase
        )),
        lateral_err_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='curvature_reference',
            std_filter_value=profiling_conf.control_metrics.curvature_harsh_limit,
            std_norm_name='lateral_error',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
                              * profiling_conf.vehicle_wheelbase
        )),
        lateral_err_rate_std=compute_std(grading_mtx, grading_arguments(
            std_filter_name='speed_reference',
            std_filter_value=profiling_conf.control_metrics.speed_stop,
            std_norm_name='lateral_error_rate',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.vehicle_wheelbase
        )),
        lateral_err_rate_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='curvature_reference',
            std_filter_value=profiling_conf.control_metrics.curvature_harsh_limit,
            std_norm_name='lateral_error_rate',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.vehicle_wheelbase
        )),
        heading_err_std=compute_std(grading_mtx, grading_arguments(
            std_filter_name='speed_reference',
            std_filter_value=profiling_conf.control_metrics.speed_stop,
            std_norm_name='heading_error',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        )),
        heading_err_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='curvature_reference',
            std_filter_value=profiling_conf.control_metrics.curvature_harsh_limit,
            std_norm_name='heading_error',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        )),
        heading_err_rate_std=compute_std(grading_mtx, grading_arguments(
            std_filter_name='speed_reference',
            std_filter_value=profiling_conf.control_metrics.speed_stop,
            std_norm_name='heading_error_rate',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=1.0
        )),
        heading_err_rate_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='curvature_reference',
            std_filter_value=profiling_conf.control_metrics.curvature_harsh_limit,
            std_norm_name='heading_error',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=1.0
        )),
        station_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='station_error',
            peak_filter_name='',
            peak_filter_value='',
            peak_threshold=profiling_conf.control_metrics.station_error_thold
        )),
        speed_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='speed_error',
            peak_filter_name='',
            peak_filter_value='',
            peak_threshold=profiling_conf.control_metrics.speed_error_thold
        )),
        lateral_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='lateral_error',
            peak_filter_name='',
            peak_filter_value='',
            peak_threshold=profiling_conf.control_metrics.lateral_error_thold
        )),
        lateral_err_rate_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='lateral_error_rate',
            peak_filter_name='',
            peak_filter_value='',
            peak_threshold=profiling_conf.control_metrics.lateral_error_rate_thold
        )),
        heading_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='heading_error',
            peak_filter_name='',
            peak_filter_value='',
            peak_threshold=profiling_conf.control_metrics.heading_error_thold
        )),
        heading_err_rate_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='heading_error_rate',
            peak_filter_name='',
            peak_filter_value='',
            peak_threshold=profiling_conf.control_metrics.heading_error_rate_thold
        )),
        ending_station_err=compute_ending(grading_mtx, grading_arguments(
            ending_feature_name='station_error',
            ending_time_name='timestamp_sec',
            ending_filter_name=['speed','path_remain'],
            ending_filter_value=[profiling_conf.control_metrics.speed_stop,
                                 profiling_conf.control_metrics.station_error_thold],
            ending_threshold=profiling_conf.control_metrics.station_error_thold
        )),
        ending_lateral_err=compute_ending(grading_mtx, grading_arguments(
            ending_feature_name='lateral_error',
            ending_time_name='timestamp_sec',
            ending_filter_name=['speed','path_remain'],
            ending_filter_value=[profiling_conf.control_metrics.speed_stop,
                                 profiling_conf.control_metrics.station_error_thold],
            ending_threshold=profiling_conf.control_metrics.lateral_error_thold
        )),
        ending_heading_err=compute_ending(grading_mtx, grading_arguments(
            ending_feature_name='heading_error',
            ending_time_name='timestamp_sec',
            ending_filter_name=['speed','path_remain'],
            ending_filter_value=[profiling_conf.control_metrics.speed_stop,
                                 profiling_conf.control_metrics.station_error_thold],
            ending_threshold=profiling_conf.control_metrics.heading_error_thold
        )),
        acc_bad_sensation=compute_beyond(grading_mtx, grading_arguments(
            beyond_feature_name='acceleration',
            beyond_threshold=profiling_conf.control_metrics.acceleration_thold
        )),
        jerk_bad_sensation=compute_beyond(grading_mtx, grading_arguments(
            beyond_feature_name='jerk',
            beyond_threshold=profiling_conf.control_metrics.jerk_thold
        )),
        lateral_acc_bad_sensation=compute_beyond(grading_mtx, grading_arguments(
            beyond_feature_name='lateral_acceleration',
            beyond_threshold=profiling_conf.control_metrics.lat_acceleration_thold
        )),
        lateral_jerk_bad_sensation=compute_beyond(grading_mtx, grading_arguments(
            beyond_feature_name='lateral_jerk',
            beyond_threshold=profiling_conf.control_metrics.lat_jerk_thold
        )),
        heading_acc_bad_sensation=compute_beyond(grading_mtx, grading_arguments(
            beyond_feature_name='heading_acceleration',
            beyond_threshold=profiling_conf.control_metrics.heading_acceleration_thold
        )),
        heading_jerk_bad_sensation=compute_beyond(grading_mtx, grading_arguments(
            beyond_feature_name='heading_jerk',
            beyond_threshold=profiling_conf.control_metrics.heading_jerk_thold
        )),
        throttle_control_usage=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='throttle_cmd',
            usage_thold_value='',
            usage_threshold='',
            usage_weight=profiling_conf.control_command_pct
        )),
        brake_control_usage=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='brake_cmd',
            usage_thold_value='',
            usage_threshold='',
            usage_weight=profiling_conf.control_command_pct
        )),
        steering_control_usage=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='steering_cmd',
            usage_thold_value='',
            usage_threshold='',
            usage_weight=profiling_conf.control_command_pct
        )),
        throttle_control_usage_harsh=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='throttle_cmd',
            usage_thold_value='acceleration_reference',
            usage_threshold=profiling_conf.control_metrics.acceleration_harsh_limit,
            usage_weight=profiling_conf.control_command_pct
        )),
        brake_control_usage_harsh=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='brake_cmd',
            usage_thold_value='acceleration_reference',
            usage_threshold=profiling_conf.control_metrics.acceleration_harsh_limit,
            usage_weight=profiling_conf.control_command_pct
        )),
        steering_control_usage_harsh=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='steering_cmd',
            usage_thold_value='curvature_reference',
            usage_threshold=profiling_conf.control_metrics.curvature_harsh_limit,
            usage_weight=profiling_conf.control_command_pct
        )),
        throttle_deadzone_mean=compute_mean(grading_mtx, grading_arguments(
            mean_feature_name='throttle_chassis',
            mean_filter_name='brake_cmd',
            mean_filter_value=feature_utils.MIN_EPSILON,
            mean_weight=profiling_conf.control_command_pct
        )),
        brake_deadzone_mean=compute_mean(grading_mtx, grading_arguments(
            mean_feature_name='brake_chassis',
            mean_filter_name='throttle_cmd',
            mean_filter_value=feature_utils.MIN_EPSILON,
            mean_weight=profiling_conf.control_command_pct
        )),
        total_time_usage=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='total_time',
            usage_thold_value='',
            usage_threshold='',
            usage_weight=profiling_conf.control_period * profiling_conf.total_time_factor
        )),
        total_time_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='total_time',
            peak_threshold=profiling_conf.control_period * profiling_conf.total_time_factor
        )),
        total_time_exceeded_count=compute_count(grading_mtx, grading_arguments(
            count_feature_name='total_time_exceeded'
        )),
        pose_heading_offset_std=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='pose_heading_offset',
            usage_thold_value='speed_reference',
            usage_threshold=profiling_conf.control_metrics.speed_stop,
            usage_weight=math.pi
        )),
        pose_heading_offset_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='pose_heading_offset',
            peak_filter_name='speed_reference',
            peak_filter_value=profiling_conf.control_metrics.speed_stop,
            peak_threshold=math.pi
        )))
    return (target, grading_group_result)


def compute_std(grading_mtx, arg):
    """Compute the std deviation value by using specified arguments in namedtuple format"""
    profiling_conf = feature_utils.get_config_control_profiling()
    if arg.std_filter_name:
        grading_mtx = filter_value(grading_mtx,
                                   FEATURE_IDX[arg.std_filter_name], arg.std_filter_value)
    elem_num, _ = grading_mtx.shape
    if elem_num < profiling_conf.min_sample_size:
        glog.warn('no enough elements {} for std computing requirement {}'
                  .format(elem_num, profiling_conf.min_sample_size))
        return (0.0, 0)
    column_norm = grading_mtx[:, FEATURE_IDX[arg.std_norm_name]]
    column_denorm = grading_mtx[:, np.array([FEATURE_IDX[denorm_name]
                                for denorm_name in arg.std_denorm_name])]
    column_denorm = np.maximum(np.fabs(column_denorm), arg.std_max_compare)
    column_denorm = [np.prod(column) for column in column_denorm]
    std = [nor / (denor * arg.std_denorm_weight)
           for nor, denor in zip(column_norm, column_denorm)]
    return (get_std_value(std), elem_num)

def compute_peak(grading_mtx, arg):
    """Compute the peak value"""
    profiling_conf = feature_utils.get_config_control_profiling()
    if arg.peak_filter_name:
        grading_mtx = filter_value(grading_mtx,
                                   FEATURE_IDX[arg.peak_filter_name], arg.peak_filter_value)
    elem_num, _ = grading_mtx.shape
    if elem_num < profiling_conf.min_sample_size:
        glog.warn('no enough elements {} for peak computing requirement {}'
                  .format(elem_num, profiling_conf.min_sample_size))
        return (0.0, 0)
    return (np.max(np.fabs(grading_mtx[:, FEATURE_IDX[arg.peak_feature_name]])) /
            arg.peak_threshold, elem_num)

def compute_ending(grading_mtx, arg):
    """Compute the specific value at the final state"""
    profiling_conf = feature_utils.get_config_control_profiling()
    if arg.ending_filter_name:
        for idx in range(len(arg.ending_filter_name)):
            grading_mtx = filter_value(grading_mtx, FEATURE_IDX[arg.ending_filter_name[idx]],
                                       arg.ending_filter_value[idx], 1)
    elem_num, item_num = grading_mtx.shape
    if elem_num < 1:
        glog.warn('no enough elements {} for ending computing requirement {}'
                  .format(elem_num, 1))
        return ([[0.0], [0.0],[0.0]], 0)
    grading_mtx = grading_mtx[np.argsort(grading_mtx[:, FEATURE_IDX[arg.ending_time_name]])]
    static_error = [np.fabs(grading_mtx[0, FEATURE_IDX[arg.ending_feature_name]]) /
                    arg.ending_threshold];
    static_start_time = [grading_mtx[0, FEATURE_IDX[arg.ending_time_name]]];
    static_stop_timne = [grading_mtx[0, FEATURE_IDX[arg.ending_time_name]]];
    for idx in range(1, grading_mtx.shape[0]):
        if (grading_mtx[idx, FEATURE_IDX[arg.ending_time_name]] -
            grading_mtx[idx-1, FEATURE_IDX[arg.ending_time_name]] <= 1.0):
            static_stop_timne[-1] = grading_mtx[idx, FEATURE_IDX[arg.ending_time_name]]
        else:
            static_error.append(np.fabs(grading_mtx[idx, FEATURE_IDX[arg.ending_feature_name]]) /
                                arg.ending_threshold)
            static_start_time.append(grading_mtx[idx, FEATURE_IDX[arg.ending_time_name]])
            static_stop_time.append(grading_mtx[idx, FEATURE_IDX[arg.ending_time_name]])
    return ([static_error, static_start_time, static_stop_timne], elem_num)

def compute_usage(grading_mtx, arg):
    """Compute the usage value"""
    profiling_conf = feature_utils.get_config_control_profiling()
    if arg.usage_thold_value:
        grading_mtx = filter_value(grading_mtx,
                                   FEATURE_IDX[arg.usage_thold_value], arg.usage_threshold)
    elem_num, _ = grading_mtx.shape
    if elem_num < profiling_conf.min_sample_size:
        glog.warn('no enough elements {} for usage computing requirement {}'
                  .format(elem_num, profiling_conf.min_sample_size))
        return (0.0, 0)
    return (get_std_value([val / arg.usage_weight
                           for val in grading_mtx[:, FEATURE_IDX[arg.usage_feature_name]]]),
            elem_num)

def compute_beyond(grading_mtx, arg):
    """Compute the beyond_the_threshold counting value"""
    elem_num, _ = grading_mtx.shape
    return (len(np.where(np.fabs(grading_mtx[:, FEATURE_IDX[arg.beyond_feature_name]]) >=
                         arg.beyond_threshold)) / elem_num,
            elem_num)

def compute_count(grading_mtx, arg):
    """Compute the event (boolean true) counting value"""
    elem_num, _ = grading_mtx.shape
    return (len(np.where(grading_mtx[:, FEATURE_IDX[arg.count_feature_name]] == 1)) / elem_num,
            elem_num)

def compute_mean(grading_mtx, arg):
    """Compute the mean value"""
    profiling_conf = feature_utils.get_config_control_profiling()
    if arg.mean_filter_name:
        grading_mtx = filter_value(grading_mtx,
                                   FEATURE_IDX[arg.mean_filter_name], arg.mean_filter_value)
    elem_num, item_num = grading_mtx.shape
    if elem_num < profiling_conf.min_sample_size:
        glog.warn('no enough elements {} for mean computing requirement {}'
                  .format(elem_num, profiling_conf.min_sample_size))
        return (0.0, 0)
    if item_num <= FEATURE_IDX[arg.mean_feature_name]:
        glog.warn('no desired feature item {} for mean computing requirement {}'
                  .format(item_num, FEATURE_IDX[arg.mean_feature_name]))
        return (0.0, 0)
    return (np.mean(grading_mtx[:, FEATURE_IDX[arg.mean_feature_name]], axis=0) / arg.mean_weight,
            elem_num)

def get_std_value(grading_column):
    """Calculate the standard deviation value"""
    return (sum(val**2 for val in grading_column) / (len(grading_column)-1)) ** 0.5

def filter_value(grading_mtx, column_name, threshold, filter_mode=0):
    """Filter the rows out if they do not satisfy threshold values"""
    if filter_mode == 0:
        return np.delete(grading_mtx,
                         np.where(np.fabs(grading_mtx[:, column_name]) < threshold), axis=0)
    if filter_mode == 1:
        return np.delete(grading_mtx,
                         np.where(np.fabs(grading_mtx[:, column_name]) >= threshold), axis=0)
    return grading_mtx

def combine_gradings(grading_x, grading_y):
    """Reduce gradings by combining the groups with different strategies"""
    # TODO (Yu): creating sub classes that inherit corresponding operations from base classes
    if not grading_x:
        return grading_y
    elif not grading_y:
        return grading_x
    grading_group_value = []
    for idx in range(len(grading_x._fields)):
        val_x, num_x = grading_x[idx]
        val_y, num_y = grading_y[idx]
        # Standard deviation and usage values
        if (grading_x._fields[idx].find('std') >= 0 or
            grading_x._fields[idx].find('usage') >= 0):
            if num_x + num_y != 0:
                grading_item_value = ((val_x ** 2 * (num_x - 1) + val_y ** 2 * (num_y - 1))
                                      / (num_x + num_y -1)) ** 0.5
            else:
                grading_item_value = 0.0
        # Peak values
        elif grading_x._fields[idx].find('peak') >= 0:
            grading_item_value = max(val_x, val_y)
        # Ending error Values
        elif grading_x._fields[idx].find('ending') >= 0:
            if num_x != 0 and num_y != 0:
                # If the first static_start_time in val_x is earlier than the one in val_y
                if val_x[1][0] < val_y[1][0]:
                    grading_item_value = val_x
                    append_value = val_y
                else:
                    grading_item_value = val_y
                    append_alue = val_x
                # If the first static_start_time in append_value is close to the last
                # static_stop_time in grading_item_value
                if append_value[1][0] - grading_item_value[2][-1] <= 1.0:
                    grading_item_value[2][-1] = append_value[2][0]
                    if len(append_value[0]) > 1:
                        grading_item_value[0].extend(append_value[0][1:])
                        grading_item_value[1].extend(append_value[1][1:])
                        grading_item_value[2].extend(append_value[2][1:])
                else:
                    grading_item_value[0].extend(append_value[0])
                    grading_item_value[1].extend(append_value[1])
                    grading_item_value[2].extend(append_value[2])
            elif num_x == 0:
                grading_item_value = val_y
            else:
                grading_item_value = val_x
        # Beyond and count values
        elif (grading_x._fields[idx].find('sensation') >= 0 or
              grading_x._fields[idx].find('count') >= 0 or
              grading_x._fields[idx].find('mean') >= 0):
            if num_x + num_y != 0:
                grading_item_value = (val_x * num_x + val_y * num_y) / (num_x + num_y)
            else:
                grading_item_value = 0.0
        grading_item_num = num_x + num_y
        grading_group_value.append((grading_item_value, grading_item_num))
    grading_x = grading_x._make(grading_group_value)
    return grading_x

def output_gradings(target_grading):
    """Write the grading results to files in coresponding target dirs"""
    target_dir, grading = target_grading
    profiling_conf = feature_utils.get_config_control_profiling()
    grading_output_path = os.path.join(target_dir,
                                       '{}_{}_control_performance_grading.txt'
                                       .format(profiling_conf.vehicle_type,
                                               profiling_conf.controller_type))
    glog.info('writing grading output {} to {}'.format(grading, grading_output_path))
    if not grading:
        glog.warn('No grading results written to {}'.format(grading_output_path))
    else:
        with open(grading_output_path, 'w') as grading_file:
            grading_file.write('Grading_output: \t {0:<32s} {1:<16s} {2:<16s} {3:<16s}\n'
                               .format('Grading Items', 'Grading Values', 'Sampling Size',
                                       'Event Timestamp'))
            for name, value in grading._asdict().iteritems():
                if not value:
                    glog.warn('grading value for {} is None'.format(name))
                    continue
                # For the ending_XXX_err values, the data are stored in multiple-dimentional list
                # in the first element of value tuples
                if isinstance(value[0], list) and 'ending_' in name:
                    for idx in range(len(value[0][0])):
                        grading_file.write('Grading_output: \t {0:<32s} {1:<16.3%} {2:<16n} {3:<16.3f} \n'
                                           .format(name + '_trajectory_' + str(idx),
                                                   value[0][0][idx], value[1], value[0][1][idx]))
                # For the other values, the data are stored as one float variable in the first
                # element of value tuples
                else:
                    grading_file.write('Grading_output: \t {0:<32s} {1:<16.3%} {2:<16n} \n'
                                       .format(name, value[0], value[1]))
            grading_file.write('\n\n\nMetrics in file control_profiling_conf.pb.txt\n\n')
            grading_file.write('{}\n\n'.format(profiling_conf))

def highlight_gradings(task, grading_file):
    """extract the highlighted information from gradings and publish them via summarize_tasks"""
    highlight_std_items = ['station_err_std',
                           'speed_err_std',
                           'lateral_err_std',
                           'heading_err_std']
    highlight_peak_items =  ['station_err_peak',
                             'speed_err_peak',
                             'lateral_err_peak',
                             'heading_err_peak',
                             'total_time_peak']
    std_scores = []
    peak_scores = []
    std_samples = []
    peak_samples = []
    if not grading_file:
        glog.warn('No grading files found under the targeted path for task: {}'.format(task))
        return ([], [])
    for file in grading_file:
        glog.info('Loading {}'.format(file))
        with open(file, 'r') as informations:
            for information in informations:
                gradings = information.split()
                if (len(gradings) > 0):
                    for idx in range(len(gradings)):
                        if gradings[idx] in highlight_std_items:
                            std_scores.append("=".join([gradings[idx], gradings[idx+1]]))
                            std_samples.append("=".join([gradings[idx], gradings[idx+2]]))
                        if gradings[idx] in highlight_peak_items:
                            peak_scores.append("=".join([gradings[idx], gradings[idx+1]]))
                            peak_samples.append("=".join([gradings[idx], gradings[idx+2]]))
            highlight_scores = std_scores + ["____________________"] + peak_scores + ["<br />"]
            highlight_samples = std_samples + ["____________________"] + peak_samples + ["<br />"]
    return (highlight_scores, highlight_samples)
