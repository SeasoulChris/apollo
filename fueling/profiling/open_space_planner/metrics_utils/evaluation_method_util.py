#!/usr/bin/env python

""" Open-space planner feature processing related utils. """

from collections import namedtuple
import os

import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from fueling.profiling.common.stats_utils import compute_stats
from fueling.profiling.conf.open_space_planner_conf import FEATURE_IDX
from fueling.profiling.proto.open_space_planner_profiling_pb2 import OpenSpacePlannerProfiling


GradingResults = namedtuple('grading_results',
                            ['end_to_end_time',
                             'zigzag_time',
                             'stage_completion_time',
                             'non_gear_switch_length_ratio',
                             'initial_heading_diff_ratio',
                             'curvature_ratio',
                             'curvature_change_ratio',
                             'acceleration_ratio',
                             'deceleration_ratio',
                             'longitudinal_acceleration_ratio',
                             'longitudinal_deceleration_ratio',
                             'lateral_acceleration_ratio',
                             'lateral_deceleration_ratio',
                             'longitudinal_positive_jerk_ratio',
                             'longitudinal_negative_jerk_ratio',
                             'lateral_positive_jerk_ratio',
                             'lateral_negative_jerk_ratio',
                             'distance_to_roi_boundaries_ratio',
                             ])
GradingResults.__new__.__defaults__ = (None,) * len(GradingResults._fields)


def get_config_open_space_profiling():
    """Get configuration from open_space_planner_profiling_conf.pb.txt"""
    profiling_conf_file = '/fuel/fueling/profiling/conf/open_space_planner_profiling_conf.pb.txt'
    open_space_planner_profiling = OpenSpacePlannerProfiling()
    proto_utils.get_pb_from_text_file(profiling_conf_file, open_space_planner_profiling)
    return open_space_planner_profiling


def stats_helper(feature_mtx, feature_name, above_threshold=True,
                 filter_name='', filter_value='', filter_mode=''):
    profiling_conf = get_config_open_space_profiling()
    return compute_stats(feature_mtx, feature_name, profiling_conf, FEATURE_IDX,
                         above_threshold=above_threshold, ratio_threshold=1.0, percentile=95,
                         filter_name=filter_name, filter_value=filter_value, filter_mode=filter_mode)


def merge_grading_results(grading_tuple):
    def find(*values):
        return next((x for x in values if x is not None), None)
    return GradingResults(*map(find, grading_tuple[0], grading_tuple[1]))


def stage_grading(target_groups):
    target, group_id, feature_mtx = target_groups
    if feature_mtx.shape[0] == 0:
        logging.warning(F'No valid element in group {group_id} for target {target}')
        return (target, None)
    if feature_mtx.shape[0] != 1:
        logging.warning(F'Unexpected number of elements in group {group_id} for target {target}'
                        'Only one element/sample should be present!')
        return (target, None)

    # TODO(shu): added scenario type and stage type
    h5_output_file = '_{:05d}_stage'.format(group_id)
    logging.info(F'Writing {feature_mtx.shape[0]} samples to h5 file {h5_output_file} '
                 F'for target {target}')
    h5_utils.write_h5(feature_mtx, target, h5_output_file)

    grading_group_result = GradingResults(
        stage_completion_time=(feature_mtx[0, FEATURE_IDX['stage_completion_time']],
                               feature_mtx.shape[0]),
        initial_heading_diff_ratio=(feature_mtx[0, FEATURE_IDX['initial_heading_diff_ratio']],
                                    feature_mtx.shape[0]),
    )
    return (target, grading_group_result)


def latency_grading(target_groups):
    target, group_id, feature_mtx = target_groups
    if feature_mtx.shape[0] == 0:
        logging.warning(F'No valid element in group {group_id} for target {target}')
        return (target, None)

    # TODO(shu): added scenario type and stage type
    h5_output_file = '_{:05d}_latency'.format(group_id)
    logging.info(F'Writing {feature_mtx.shape[0]} samples to h5 file {h5_output_file} '
                 F'for target {target}')
    h5_utils.write_h5(feature_mtx, target, h5_output_file)

    grading_group_result = GradingResults(
        # Exclude HitBoundTimes for these metrics
        end_to_end_time=stats_helper(feature_mtx, 'end_to_end_time')[1:],
        zigzag_time=stats_helper(feature_mtx, 'zigzag_time', True, filter_name=['zigzag_time'],
                                 filter_value=[0.1], filter_mode=[0])[1:],
    )
    return (target, grading_group_result)


def zigzag_grading(target_groups):
    target, group_id, feature_mtx = target_groups
    if feature_mtx.shape[0] == 0:
        logging.warning(F'No valid element in group {group_id} for target {target}')
        return (target, None)

    # TODO(shu): added scenario type and stage type
    h5_output_file = '_{:05d}_zigzag'.format(group_id)
    logging.info(F'Writing {feature_mtx.shape[0]} samples to h5 file {h5_output_file} '
                 F'for target {target}')
    h5_utils.write_h5(feature_mtx, target, h5_output_file)

    grading_group_result = GradingResults(
        non_gear_switch_length_ratio=stats_helper(feature_mtx, 'non_gear_switch_length_ratio',
                                                  True, filter_name=[
                                                      'non_gear_switch_length_ratio'],
                                                  filter_value=[0.0], filter_mode=[0]),
    )
    return (target, grading_group_result)


def trajectory_grading(target_groups):
    target, group_id, feature_mtx = target_groups
    if feature_mtx.shape[0] == 0:
        logging.warning(F'No valid element in group {group_id} for target {target}')
        return (target, None)

    # TODO(shu): added scenario type and stage type
    h5_output_file = '_{:05d}_trajectory'.format(group_id)
    logging.info(F'Writing {feature_mtx.shape[0]} samples to h5 file {h5_output_file} '
                 F'for target {target}')
    h5_utils.write_h5(feature_mtx, target, h5_output_file)

    grading_group_result = GradingResults(
        curvature_ratio=stats_helper(feature_mtx, 'curvature_ratio'),
        curvature_change_ratio=stats_helper(feature_mtx, 'curvature_change_ratio'),
        acceleration_ratio=stats_helper(feature_mtx, 'acceleration_ratio'),
        deceleration_ratio=stats_helper(feature_mtx, 'deceleration_ratio'),

        longitudinal_acceleration_ratio=stats_helper(
            feature_mtx, 'longitudinal_acceleration_ratio', True),
        longitudinal_deceleration_ratio=stats_helper(
            feature_mtx, 'longitudinal_deceleration_ratio', False),
        lateral_acceleration_ratio=stats_helper(feature_mtx, 'lateral_acceleration_ratio', True),
        lateral_deceleration_ratio=stats_helper(feature_mtx, 'lateral_deceleration_ratio', False),

        longitudinal_positive_jerk_ratio=stats_helper(
            feature_mtx, 'longitudinal_positive_jerk_ratio'),
        longitudinal_negative_jerk_ratio=stats_helper(
            feature_mtx, 'longitudinal_negative_jerk_ratio'),
        lateral_positive_jerk_ratio=stats_helper(feature_mtx, 'lateral_positive_jerk_ratio'),
        lateral_negative_jerk_ratio=stats_helper(feature_mtx, 'lateral_negative_jerk_ratio'),
        distance_to_roi_boundaries_ratio=stats_helper(
            feature_mtx, 'distance_to_roi_boundaries_ratio'),
    )
    return (target, grading_group_result)


def output_result(target_grading):
    """Write the grading results to files in corresponding target dirs"""
    target_dir, grading = target_grading
    grading_output_path = os.path.join(target_dir, 'open_space_performance_grading.txt')
    logging.info(F'Writing grading output {grading} to {target_dir}')

    grading_dict = grading._asdict()
    with open(grading_output_path, 'w') as grading_file:
        grading_file.write('{:<36s} {:<16s} {:<16s} {:<16s} {:<16s} {:<16s} {:<16s} {:<16s}\n'
                           .format('Metric', 'HitBoundTimes', 'Max', 'Mean', 'StdDev',
                                   '95Percentile', 'SingleValue', 'SampleSize'))
        for name, value in grading_dict.items():
            if not value:
                logging.warning(F'Grading result for {name} is None')
                continue

            if len(value) == 2:
                # it has a single value and num of elements
                grading_file.write('{:<36s} {:<84s} {:<16.6f} {:<16n}\n'
                                   .format(name, ' ', value[0], value[-1]))
            if len(value) == 5:
                # it has 4 statistics and num of elements
                grading_file.write('{:<36s} {:<16s} {:<16.3f} {:<16.3f} {:<16.3f} {:<16.3f} {:<16s} {:<16n}\n'
                                   .format(name, 'N.A.', value[0], value[1], value[2], value[3],
                                           'N.A.', value[-1]))
            if len(value) == 6:
                # it has 5 statistics and num of elements
                grading_file.write('{:<36s} {:<16n} {:<16.3%} {:<16.3%} {:<16.3%} {:<16.3%} {:<16s} {:<16n}\n'
                                   .format(name, value[0], value[1], value[2], value[3], value[4],
                                           'N.A.', value[-1]))
