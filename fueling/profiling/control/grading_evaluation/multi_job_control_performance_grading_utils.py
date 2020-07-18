#!/usr/bin/env python
""" Control performance grading related utils. """

from collections import namedtuple
import json
import math
import os

from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, WEIGHTED_SCORE
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.redis_utils as redis_utils
from fueling.profiling.common.stats_utils import compute_beyond, compute_count, compute_ending, \
    compute_mean, compute_peak, compute_rms, compute_usage, GradingArguments
import fueling.profiling.control.feature_extraction.multi_job_control_feature_extraction_utils \
    as feature_utils
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils


def compute_h5_and_gradings(target_groups, flags):
    """Do computing against one group"""
    target, group_id, msgs = target_groups
    logging.info(F'computing {len(msgs)} messages for target {target}')
    profiling_conf = feature_utils.get_config_control_profiling()
    if flags['ctl_metrics_simulation_only_test']:
        vehicle_param = multi_vehicle_utils.get_vehicle_param(target)
    else:
        vehicle_param = multi_vehicle_utils.get_vehicle_param_by_target(target)
    grading_mtx = feature_utils.extract_data_at_multi_channels(msgs, flags,
                                                               profiling_conf.driving_mode,
                                                               profiling_conf.gear_position,
                                                               profiling_conf.control_error_code)
    if grading_mtx.shape[0] == 0:
        logging.warning(F'no valid element in {len(msgs)} items in group {group_id}'
                        F'for task {target}')
        return (target, None)

    if flags['ctl_metrics_save_report']:
        h5_output_file = '{:05d}'.format(group_id)
        logging.info(F'writing {grading_mtx.shape[0]} messages ({grading_mtx.shape[1]} dimensions) '
                     F'to h5 file {h5_output_file} for target {target}')
        h5_utils.write_h5(grading_mtx, target, h5_output_file)
    else:
        file_utils.makedirs(target)

    GradingResults = namedtuple('grading_results',
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
                                 'replan_trajectory_count',
                                 'pose_heading_offset_std',
                                 'pose_heading_offset_peak',
                                 'control_error_code_count'])
    GradingResults.__new__.__defaults__ = (None,) * len(GradingResults._fields)

    grading_group_result = GradingResults(
        station_err_std=compute_rms(grading_mtx, GradingArguments(
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            norm_name='station_error',
            denorm_name=['speed_reference'],
            max_compare=[profiling_conf.control_metrics.speed_still],
            denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        station_err_std_harsh=compute_rms(grading_mtx, GradingArguments(
            filter_name=['acceleration_reference'],
            filter_value=[
                profiling_conf.control_metrics.acceleration_harsh_limit],
            filter_mode=[0],
            norm_name='station_error',
            denorm_name=['speed_reference'],
            max_compare=[profiling_conf.control_metrics.speed_still],
            denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        speed_err_std=compute_rms(grading_mtx, GradingArguments(
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            norm_name='speed_error',
            denorm_name=['speed_reference'],
            max_compare=[profiling_conf.control_metrics.speed_still],
            denorm_weight=1.0
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        speed_err_std_harsh=compute_rms(grading_mtx, GradingArguments(
            filter_name=['acceleration_reference'],
            filter_value=[
                profiling_conf.control_metrics.acceleration_harsh_limit],
            filter_mode=[0],
            norm_name='speed_error',
            denorm_name=['speed_reference'],
            max_compare=[profiling_conf.control_metrics.speed_still],
            denorm_weight=1.0
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        lateral_err_std=compute_rms(grading_mtx, GradingArguments(
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            norm_name='lateral_error',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
            * vehicle_param.wheel_base
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        lateral_err_std_harsh=compute_rms(grading_mtx, GradingArguments(
            filter_name=['curvature_reference'],
            filter_value=[
                profiling_conf.control_metrics.curvature_harsh_limit],
            filter_mode=[0],
            norm_name='lateral_error',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
            * vehicle_param.wheel_base
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        lateral_err_rate_std=compute_rms(grading_mtx, GradingArguments(
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            norm_name='lateral_error_rate',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=vehicle_param.wheel_base
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        lateral_err_rate_std_harsh=compute_rms(grading_mtx, GradingArguments(
            filter_name=['curvature_reference'],
            filter_value=[
                profiling_conf.control_metrics.curvature_harsh_limit],
            filter_mode=[0],
            norm_name='lateral_error_rate',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=vehicle_param.wheel_base
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        heading_err_std=compute_rms(grading_mtx, GradingArguments(
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            norm_name='heading_error',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        heading_err_std_harsh=compute_rms(grading_mtx, GradingArguments(
            filter_name=['curvature_reference'],
            filter_value=[
                profiling_conf.control_metrics.curvature_harsh_limit],
            filter_mode=[0],
            norm_name='heading_error',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=profiling_conf.control_period * profiling_conf.control_frame_num
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        heading_err_rate_std=compute_rms(grading_mtx, GradingArguments(
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            norm_name='heading_error_rate',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=1.0
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        heading_err_rate_std_harsh=compute_rms(grading_mtx, GradingArguments(
            filter_name=['curvature_reference'],
            filter_value=[
                profiling_conf.control_metrics.curvature_harsh_limit],
            filter_mode=[0],
            norm_name='heading_error',
            denorm_name=['curvature_reference', 'speed_reference'],
            max_compare=[profiling_conf.control_metrics.curvature_still,
                         profiling_conf.control_metrics.speed_still],
            denorm_weight=1.0
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        station_err_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='station_error',
            time_name='timestamp_sec',
            filter_name='',
            filter_value='',
            threshold=profiling_conf.control_metrics.station_error_thold
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        speed_err_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='speed_error',
            time_name='timestamp_sec',
            filter_name='',
            filter_value='',
            threshold=profiling_conf.control_metrics.speed_error_thold
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        lateral_err_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='lateral_error',
            time_name='timestamp_sec',
            filter_name='',
            filter_value='',
            threshold=profiling_conf.control_metrics.lateral_error_thold
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        lateral_err_rate_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='lateral_error_rate',
            time_name='timestamp_sec',
            filter_name='',
            filter_value='',
            threshold=profiling_conf.control_metrics.lateral_error_rate_thold
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        heading_err_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='heading_error',
            time_name='timestamp_sec',
            filter_name='',
            filter_value='',
            threshold=profiling_conf.control_metrics.heading_error_thold
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        heading_err_rate_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='heading_error_rate',
            time_name='timestamp_sec',
            filter_name='',
            filter_value='',
            threshold=profiling_conf.control_metrics.heading_error_rate_thold
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        ending_station_err=compute_ending(grading_mtx, GradingArguments(
            feature_name='station_error',
            time_name='timestamp_sec',
            filter_name=['speed', 'path_remain'],
            filter_value=[profiling_conf.control_metrics.speed_stop,
                          profiling_conf.control_metrics.station_error_thold],
            filter_mode=[1, 1],
            threshold=profiling_conf.control_metrics.station_error_thold
        ), FEATURE_IDX),
        ending_lateral_err=compute_ending(grading_mtx, GradingArguments(
            feature_name='lateral_error',
            time_name='timestamp_sec',
            filter_name=['speed', 'path_remain'],
            filter_value=[profiling_conf.control_metrics.speed_stop,
                          profiling_conf.control_metrics.station_error_thold],
            filter_mode=[1, 1],
            threshold=profiling_conf.control_metrics.lateral_error_thold
        ), FEATURE_IDX),
        ending_heading_err=compute_ending(grading_mtx, GradingArguments(
            feature_name='heading_error',
            time_name='timestamp_sec',
            filter_name=['speed', 'path_remain'],
            filter_value=[profiling_conf.control_metrics.speed_stop,
                          profiling_conf.control_metrics.station_error_thold],
            filter_mode=[1, 1],
            threshold=profiling_conf.control_metrics.heading_error_thold
        ), FEATURE_IDX),
        acc_bad_sensation=compute_beyond(grading_mtx, GradingArguments(
            feature_name='acceleration',
            threshold=profiling_conf.control_metrics.acceleration_thold
        ), FEATURE_IDX),
        jerk_bad_sensation=compute_beyond(grading_mtx, GradingArguments(
            feature_name='jerk',
            threshold=profiling_conf.control_metrics.jerk_thold
        ), FEATURE_IDX),
        lateral_acc_bad_sensation=compute_beyond(grading_mtx, GradingArguments(
            feature_name='lateral_acceleration',
            threshold=profiling_conf.control_metrics.lat_acceleration_thold
        ), FEATURE_IDX),
        lateral_jerk_bad_sensation=compute_beyond(grading_mtx, GradingArguments(
            feature_name='lateral_jerk',
            threshold=profiling_conf.control_metrics.lat_jerk_thold
        ), FEATURE_IDX),
        heading_acc_bad_sensation=compute_beyond(grading_mtx, GradingArguments(
            feature_name='heading_acceleration',
            threshold=profiling_conf.control_metrics.heading_acceleration_thold
        ), FEATURE_IDX),
        heading_jerk_bad_sensation=compute_beyond(grading_mtx, GradingArguments(
            feature_name='heading_jerk',
            threshold=profiling_conf.control_metrics.heading_jerk_thold
        ), FEATURE_IDX),
        throttle_control_usage=compute_usage(grading_mtx, GradingArguments(
            feature_name='throttle_cmd',
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        brake_control_usage=compute_usage(grading_mtx, GradingArguments(
            feature_name='brake_cmd',
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        steering_control_usage=compute_usage(grading_mtx, GradingArguments(
            feature_name='steering_cmd',
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        throttle_control_usage_harsh=compute_usage(grading_mtx, GradingArguments(
            feature_name='throttle_cmd',
            filter_name=['acceleration_reference'],
            filter_value=[
                profiling_conf.control_metrics.acceleration_harsh_limit],
            filter_mode=[0],
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        brake_control_usage_harsh=compute_usage(grading_mtx, GradingArguments(
            feature_name='brake_cmd',
            filter_name=['acceleration_reference'],
            filter_value=[
                profiling_conf.control_metrics.acceleration_harsh_limit],
            filter_mode=[0],
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        steering_control_usage_harsh=compute_usage(grading_mtx, GradingArguments(
            feature_name='steering_cmd',
            filter_name=['curvature_reference'],
            filter_value=[
                profiling_conf.control_metrics.curvature_harsh_limit],
            filter_mode=[0],
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        throttle_deadzone_mean=compute_mean(grading_mtx, GradingArguments(
            feature_name='throttle_chassis',
            filter_name=['brake_cmd'],
            filter_value=[feature_utils.MIN_EPSILON],
            filter_mode=[0],
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        brake_deadzone_mean=compute_mean(grading_mtx, GradingArguments(
            feature_name='brake_chassis',
            filter_name=['throttle_cmd'],
            filter_value=[feature_utils.MIN_EPSILON],
            filter_mode=[0],
            weight=profiling_conf.control_command_pct
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        total_time_usage=compute_usage(grading_mtx, GradingArguments(
            feature_name='total_time',
            weight=profiling_conf.control_period * profiling_conf.total_time_factor
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        total_time_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='total_time',
            time_name='timestamp_sec',
            threshold=profiling_conf.control_period * profiling_conf.total_time_factor
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        total_time_exceeded_count=compute_count(grading_mtx, GradingArguments(
            feature_name='total_time_exceeded'
        ), FEATURE_IDX),
        replan_trajectory_count=compute_count(grading_mtx, GradingArguments(
            feature_name='replan_flag'
        ), FEATURE_IDX),
        pose_heading_offset_std=compute_usage(grading_mtx, GradingArguments(
            feature_name='pose_heading_offset',
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            weight=math.pi
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        pose_heading_offset_peak=compute_peak(grading_mtx, GradingArguments(
            feature_name='pose_heading_offset',
            time_name='timestamp_sec',
            filter_name=['speed_reference'],
            filter_value=[profiling_conf.control_metrics.speed_stop],
            filter_mode=[0],
            threshold=math.pi
        ), profiling_conf.min_sample_size, FEATURE_IDX),
        control_error_code_count=compute_beyond(grading_mtx, GradingArguments(
            feature_name='control_error_code',
            threshold=profiling_conf.control_metrics.control_error_thold,
        ), FEATURE_IDX))
    return (target, grading_group_result)


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
        if (grading_x._fields[idx].find('std') >= 0
                or grading_x._fields[idx].find('usage') >= 0):
            if num_x + num_y != 0:
                grading_item_value = ((val_x ** 2 * (num_x - 1) + val_y ** 2 * (num_y - 1))
                                      / (num_x + num_y - 1)) ** 0.5
            else:
                grading_item_value = 0.0
        # Peak values
        elif grading_x._fields[idx].find('peak') >= 0:
            if val_x[0] >= val_y[0]:
                grading_item_value = val_x
            else:
                grading_item_value = val_y
        # Ending error Values
        elif grading_x._fields[idx].find('ending') >= 0:
            if num_x != 0 and num_y != 0:
                # If the first static_start_time in val_x is earlier than the one in val_y
                if val_x[1][0] < val_y[1][0]:
                    grading_item_value = val_x
                    append_value = val_y
                else:
                    grading_item_value = val_y
                    append_value = val_x
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
        elif (grading_x._fields[idx].find('sensation') >= 0
              or grading_x._fields[idx].find('count') >= 0
              or grading_x._fields[idx].find('mean') >= 0):
            if num_x + num_y != 0:
                grading_item_value = (
                    val_x * num_x + val_y * num_y) / (num_x + num_y)
            else:
                grading_item_value = 0.0
        grading_item_num = num_x + num_y
        grading_group_value.append((grading_item_value, grading_item_num))
    grading_x = grading_x._make(grading_group_value)
    return grading_x


def output_gradings(target_grading, flags):
    """Write the grading results to files in coresponding target dirs"""
    target_dir, grading = target_grading
    # get copied vehicle parameter conf
    if flags['ctl_metrics_simulation_only_test']:
        vehicle_type = flags['ctl_metrics_simulation_vehicle']
        controller_type = 'Sim_Controller'
    else:
        vehicle_type = multi_vehicle_utils.get_vehicle_by_target(target_dir)
        controller_type = multi_vehicle_utils.get_controller_by_target(
            target_dir)
    # Get control prifiling conf
    profiling_conf = feature_utils.get_config_control_profiling()
    grading_output_path = os.path.join(target_dir,
                                       F'{vehicle_type}_{controller_type}_'
                                       F'control_performance_grading.txt')
    profiling_conf_output_path = os.path.join(target_dir,
                                              F'{vehicle_type}_{controller_type}_'
                                              F'control_profiling_conf.pb.txt')
    if not grading:
        logging.warning(F'No grading results written to {grading_output_path}')
    else:
        grading_dict = grading._asdict()
        score = 0.0
        weights = 0.0
        sample = grading_dict['total_time_usage'][1]
        # Parse and compute the weighting metrics from control profiling results
        weighting_dict = WEIGHTED_SCORE['weighting_metrics']
        for key, weighting in weighting_dict.items():
            if 'peak' in key:
                score += grading_dict[key][0][0] * weighting
            else:
                score += grading_dict[key][0] * weighting
            weights += weighting
        score /= weights
        # Parse and compute the penalty metrics from control profiling results
        penalty_dict = WEIGHTED_SCORE['penalty_metrics']
        for key, penalty_score in penalty_dict.items():
            score += grading_dict[key][0] * \
                grading_dict[key][1] * penalty_score
        # Parse and compute the fail metrics from control profiling results
        fail_dict = WEIGHTED_SCORE['fail_metrics']
        for key, fail_score in fail_dict.items():
            if grading_dict[key][0] > 0:
                score = fail_score
        grading_dict.update({'weighted_score': (score, sample)})
        logging.info(
            F'writing grading output {grading_dict} to {grading_output_path}')
        with open(grading_output_path, 'w') as grading_file:
            grading_file.write('Grading_output: \t {0:<36s} {1:<16s} {2:<16s} {3:<16s}\n'
                               .format('Grading Items', 'Grading Values', 'Sampling Size',
                                       'Event Timestamp'))
            for name, value in grading_dict.items():
                if not value:
                    logging.warning(F'grading value for {name} is None')
                    continue
                # For the ending_XXX values and XXX_peak values, the data are stored in
                # multiple-dimentional list in the first element of value tuples
                if isinstance(value[0], list):
                    if 'ending' in name:
                        for idx in range(len(value[0][0])):
                            grading_file.write('Grading_output: \t '
                                               + '{0:<36s} {1:<16.3%} {2:<16n} {3:<16.3f} \n'
                                               .format(name + '_trajectory_' + str(idx),
                                                       value[0][0][idx], value[1],
                                                       value[0][1][idx]))
                    if 'peak' in name:
                        grading_file.write('Grading_output: \t '
                                           + '{0:<36s} {1:<16.3%} {2:<16n} {3:<16.3f} \n'
                                           .format(name, value[0][0], value[1], value[0][1]))
                # For the other values, the data are stored as one float variable in the first
                # element of value tuples
                else:
                    grading_file.write('Grading_output: \t {0:<36s} {1:<16.3%} {2:<16n} \n'
                                       .format(name, value[0], value[1]))
        with open(grading_output_path.replace('.txt', '.json'), 'w') as grading_json:
            grading_json.write(json.dumps(grading_dict))
        with open(profiling_conf_output_path, 'w') as profiling_conf_file:
            profiling_conf_file.write(F'{profiling_conf}')


def highlight_gradings(task, grading_file):
    """extract the highlighted information from gradings and publish them via summarize_tasks"""
    highlight_std_items = ['station_err_std',
                           'speed_err_std',
                           'lateral_err_std',
                           'heading_err_std']
    highlight_peak_items = ['station_err_peak',
                            'speed_err_peak',
                            'lateral_err_peak',
                            'heading_err_peak',
                            'total_time_peak']

    std_scores = []
    peak_scores = []
    std_samples = []
    peak_samples = []

    profiling_conf = feature_utils.get_config_control_profiling()
    control_redis_prefix = 'control.profiling.{}.{}'.format(
        profiling_conf.vehicle_type, profiling_conf.controller_type)

    if not grading_file:
        logging.warning(
            F'No grading files found under the targeted path for task: {task}')
        return ([], [])
    for file in grading_file:
        logging.info(F'Loading {file}')
        with open(file, 'r') as informations:
            for information in informations:
                gradings = information.split()
                if (len(gradings) > 0):
                    for idx in range(len(gradings)):
                        if not gradings[idx] in (highlight_std_items + highlight_peak_items):
                            continue

                        # Store the mapping of task->score in Redis
                        cur_redis_prefix = '{}.{}'.format(
                            control_redis_prefix, gradings[idx])
                        cur_redis_mapping = {os.path.basename(
                            task): gradings[idx + 1].strip('%')}
                        redis_utils.redis_extend_dict(
                            cur_redis_prefix, cur_redis_mapping)

                        cur_score = "=".join(
                            [gradings[idx], gradings[idx + 1]])
                        cur_sample = "=".join(
                            [gradings[idx], gradings[idx + 2]])
                        if gradings[idx] in highlight_std_items:
                            std_scores.append(cur_score)
                            std_samples.append(cur_sample)
                        else:
                            peak_scores.append(cur_score)
                            peak_samples.append(cur_sample)

            highlight_scores = std_scores + \
                ["____________________"] + peak_scores + ["<br />"]
            highlight_samples = std_samples + \
                ["____________________"] + peak_samples + ["<br />"]
    return (highlight_scores, highlight_samples)
