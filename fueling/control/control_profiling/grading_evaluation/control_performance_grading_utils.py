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
                                 ['lon_station_err_std',
                                  'lon_station_err_std_harsh',
                                  'lon_speed_err_std',
                                  'lon_speed_err_std_harsh',
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
                                  'lon_acc_bad_sensation',
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
        lon_station_err_std=compute_std(grading_mtx, grading_arguments(
            std_filter_name='',
            std_filter_value='',
            std_norm_name='station_error',
            std_denorm_name='speed_reference',
            std_max_compare=profiling_conf.control_metrics.speed_still,
            std_denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        )),
        heading_err_std_harsh=compute_std(grading_mtx, grading_arguments(
            std_filter_name='curvature_reference',
            std_filter_value=profiling_conf.control_metrics.curvature_harsh_limit,
            std_norm_name='heading_error',
            std_denorm_name='heading_reference',
            std_max_compare=profiling_conf.control_metrics.heading_still,
            std_denorm_weight=1.0
        )),
        station_err_peak=compute_peak(grading_mtx, grading_arguments(
            peak_feature_name='station_error',
            peak_threshold=profiling_conf.control_metrics.station_error_thold
        )),
        lon_acc_bad_sensation=compute_beyond(grading_mtx, grading_arguments(
            beyond_feature_name='acceleration_cmd',
            beyond_threshold=profiling_conf.control_metrics.lon_acceleration_thold
        )),
        steering_control_usage_harsh=compute_usage(grading_mtx, grading_arguments(
            usage_feature_name='acceleration_cmd',
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
    column_denorm = grading_mtx[:, feature_idx[arg.std_denorm_name]]
    std = [nor / (max(denor, arg.std_max_compare) * arg.std_denorm_weight)
           for nor, denor in zip(column_norm, column_denorm)]
    return (get_std_value(std), elem_num)

def compute_peak(grading_mtx, arg):
    """Compute the peak value"""
    elem_num, _ = grading_mtx.shape
    return (np.max(abs(grading_mtx[:, feature_idx[arg.peak_feature_name]])) / arg.peak_threshold,
            elem_num)

def compute_usage(grading_mtx, arg):
    """Compute the usage value"""
    profiling_conf = feature_utils.get_config_control_profiling()
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
                         arg.beyond_threshold)),
            elem_num)

def get_std_value(grading_column):
    """Calculate the standard deviation value"""
    return sum(val**2 for val in grading_column) / len(grading_column) ** 0.5

def filter_value(grading_mtx, column_name, threshold):
    """Filter the rows out if they do not satisfy threshold values"""
    return np.delete(grading_mtx, 
                     np.where(grading_mtx[:, column_name] < threshold), axis=0)

def combine_gradings(grading_x, grading_y):
    """Reduce gradings by combining the groups with different strategies"""
    if not grading_x:
        return grading_y
    elif not grading_y:
        return grading_x
    # Peak values
    peak_x, num_x = grading_x.station_err_peak
    peak_y, num_y = grading_y.station_err_peak
    grading_x = grading_x._replace(
        station_err_peak=(max(peak_x, peak_y), num_x + num_y))
    # Beyond values
    beyond_x, num_x = grading_x.lon_acc_bad_sensation
    beyond_y, num_y = grading_y.lon_acc_bad_sensation
    grading_x = grading_x._replace(
        lon_acc_bad_sensation=(beyond_x + beyond_y, num_x + num_y))
    # Usage and standard deviation values
    std_x, num_x = grading_x.heading_err_std_harsh
    std_y, num_y = grading_y.heading_err_std_harsh
    std_combined = 0.0
    if num_x + num_y != 0:
        std_combined = ((std_x ** 2 * (num_x - 1) + std_y ** 2 * (num_y - 1)) 
                        / (num_x + num_y)) ** 0.5
    grading_x = grading_x._replace(
        heading_err_std_harsh=(std_combined, num_x + num_y))
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
    file_utils.touch(os.path.join(target_dir, 'COMPLETE'))
