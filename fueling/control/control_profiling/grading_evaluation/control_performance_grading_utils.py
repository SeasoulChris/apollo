#!/usr/bin/env python

""" Control performance grading related utils. """

from collections import namedtuple
import os

import colored_glog as glog
import numpy as np

import fueling.common.file_utils as file_utils
import fueling.common.h5_utils as h5_utils
from fueling.control.control_profiling.conf.control_channel_conf import FEATURE_INDEX \
     as feature_idx
import fueling.control.control_profiling.feature_extraction.control_feature_extraction_utils \
       as feature_utils


# Message number in each segment
MSG_PER_SEGMENT = 1000


def compute_h5_and_gradings(target_groups):
    """Do computing against one group"""
    target, group_id, msgs = target_groups
    if len(msgs) < MSG_PER_SEGMENT:
        glog.warn('no enough items {} in group {} for task {}'.format(len(msgs), group_id, target))
        return (target, None)
    glog.info('computing {} messages for target {}'.format(len(msgs), target))
    profiling_conf = feature_utils.get_config_control_profiling()
    grading_mtx = np.array([feature_utils.extract_data_from_msg(msg) for msg in msgs])
    h5_output_file = '{}_{}_{:05d}'.format(profiling_conf.vehicle_type,
                                           profiling_conf.controller_type,
                                           group_id)
    glog.info('writing {} messages to h5 file {} for target {}'
              .format(len(msgs), h5_output_file, target))
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
                                  'steering_control_usage_harsh'])
    grading_arguments = namedtuple('grading_arguments',
                                   ['std_filter_name',
                                    'std_filter_value',
                                    'std_norm_name',
                                    'std_denorm_name',
                                    'std_max_compare',
                                    'std_denorm_weight',
                                    'peak_feature_name',
                                    'peak_threshold',
                                    'usage_feature_name',
                                    'usage_thold_value',
                                    'usage_threshold',
                                    'beyond_feature_name',
                                    'beyond_threshold'])
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
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num \
                              * profiling_conf.vehicle_wheelbase
        )),
        lateral_err_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='curvature_reference',
            std_filter_value=profiling_conf.control_metrics.curvature_harsh_limit,
            std_norm_name='lateral_error',
            std_denorm_name=['curvature_reference', 'speed_reference'],
            std_max_compare=[profiling_conf.control_metrics.curvature_still,
                             profiling_conf.control_metrics.speed_still],
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num \
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
            peak_threshold=profiling_conf.control_metrics.station_error_thold
        )),
        speed_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='speed_error',
            peak_threshold=profiling_conf.control_metrics.speed_error_thold
        )),
        lateral_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='lateral_error',
            peak_threshold=profiling_conf.control_metrics.lateral_error_thold
        )),
        lateral_err_rate_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='lateral_error_rate',
            peak_threshold=profiling_conf.control_metrics.lateral_error_rate_thold
        )),
        heading_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='heading_error',
            peak_threshold=profiling_conf.control_metrics.heading_error_thold
        )),
        heading_err_rate_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='heading_error_rate',
            peak_threshold=profiling_conf.control_metrics.heading_error_rate_thold
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
            usage_threshold=''
        )),
        brake_control_usage=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='brake_cmd',
            usage_thold_value='',
            usage_threshold=''
        )),
        steering_control_usage=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='steering_cmd',
            usage_thold_value='',
            usage_threshold=''
        )),
        throttle_control_usage_harsh=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='throttle_cmd',
            usage_thold_value='acceleration_reference',
            usage_threshold=profiling_conf.control_metrics.acceleration_harsh_limit
        )),
        brake_control_usage_harsh=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='brake_cmd',
            usage_thold_value='acceleration_reference',
            usage_threshold=profiling_conf.control_metrics.acceleration_harsh_limit
        )),
        steering_control_usage_harsh=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='steering_cmd',
            usage_thold_value='curvature_reference',
            usage_threshold=profiling_conf.control_metrics.curvature_harsh_limit
        )))
    return (target, grading_group_result)

def compute_std(grading_mtx, arg):
    """Compute the std deviation value by using specified arguments in namedtuple format"""
    profiling_conf = feature_utils.get_config_control_profiling()
    if arg.std_filter_name:
        grading_mtx = filter_value(grading_mtx,
                                   feature_idx[arg.std_filter_name], arg.std_filter_value)
    elem_num, _ = grading_mtx.shape
    if elem_num < profiling_conf.min_sample_size:
        glog.warn('no enough elements {} for std computing requirement {}'
                  .format(elem_num, profiling_conf.min_sample_size))
        return (0.0, 0)
    column_norm = grading_mtx[:, feature_idx[arg.std_norm_name]]
    column_denorm = grading_mtx[:, np.array([feature_idx[denorm_name]
                                for denorm_name in arg.std_denorm_name])]
    column_denorm = np.maximum(column_denorm,  arg.std_max_compare)
    column_denorm = [np.prod(column) for column in column_denorm]
    std = [nor / (denor * arg.std_denorm_weight)
           for nor, denor in zip(column_norm, column_denorm)]
    return (get_std_value(std), elem_num)

def compute_peak(grading_mtx, arg):
    """Compute the peak value"""
    elem_num, _ = grading_mtx.shape
    return (np.max(np.fabs(grading_mtx[:, feature_idx[arg.peak_feature_name]])) / arg.peak_threshold,
            elem_num)

def compute_usage(grading_mtx, arg):
    """Compute the usage value"""
    profiling_conf = feature_utils.get_config_control_profiling()
    if arg.usage_thold_value:
        grading_mtx = filter_value(grading_mtx,
                                   feature_idx[arg.usage_thold_value], arg.usage_threshold)
    elem_num, _ = grading_mtx.shape
    if elem_num < profiling_conf.min_sample_size:
        glog.warn('no enough elements {} for usage computing requirement {}'
                  .format(elem_num, profiling_conf.min_sample_size))
        return (0.0, 0)
    return (get_std_value([val / profiling_conf.control_command_pct
                           for val in grading_mtx[:, feature_idx[arg.usage_feature_name]]]),
            elem_num)

def compute_beyond(grading_mtx, arg):
    """Compute the usage value"""
    elem_num, _ = grading_mtx.shape
    return (len(np.where(grading_mtx[:, feature_idx[arg.beyond_feature_name]] >=
                         arg.beyond_threshold)) / elem_num,
            elem_num)

def get_std_value(grading_column):
    """Calculate the standard deviation value"""
    return (sum(val**2 for val in grading_column) / (len(grading_column)-1)) ** 0.5

def filter_value(grading_mtx, column_name, threshold):
    """Filter the rows out if they do not satisfy threshold values"""
    return np.delete(grading_mtx,
                     np.where(grading_mtx[:, column_name] < threshold), axis=0)

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
        if grading_x._fields[idx].find('std') >= 0 or grading_x._fields[idx].find('usage') >= 0:
            if num_x + num_y != 0:
                grading_item_value = ((val_x ** 2 * (num_x - 1) + val_y ** 2 * (num_y - 1))
                                      / (num_x + num_y -1)) ** 0.5
            else:
                grading_item_value = 0.0
        # Peak values
        elif grading_x._fields[idx].find('peak') >= 0:
            grading_item_value = max(val_x, val_y)
        # Beyond values
        elif grading_x._fields[idx].find('sensation') >= 0:
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
    with open(grading_output_path, 'w') as grading_file:
        grading_file.write('Grading_output: \t {0:<32s} {1:<16s} {2:<16s} \n'
                           .format('Grading Items', 'Grading Values', 'Sampling Size'))
        for name, value in grading._asdict().iteritems():
            if not value:
                glog.warn('grading value for {} is None'.format(name))
                continue
            grading_file.write('Grading_output: \t {0:<32s} {1:<16,.3%} {2:<16n} \n'
                               .format(name, value[0], value[1]))
        grading_file.write('\n\n\nMetrics in file control_profiling_conf.pb.txt\n\n')
        grading_file.write('{}\n\n'.format(profiling_conf))
    file_utils.touch(os.path.join(target_dir, 'COMPLETE'))
