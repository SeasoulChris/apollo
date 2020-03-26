#!/usr/bin/env python

""" Open-space planner feature processing related utils. """

from collections import namedtuple
import os

import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
from fueling.profiling.common.stats_utils import compute_count, compute_mean, compute_std
from fueling.profiling.conf.open_space_planner_conf import FEATURE_IDX
import fueling.profiling.open_space_planner.feature_extraction.feature_extraction_utils as feature_utils


def grading(target_groups):
    target, group_id, grading_mtx = target_groups
    if grading_mtx.shape[0] == 0:
        logging.warning('no valid element in {} items in group {} for task {}'
                        .format(grading_mtx.shape[0], group_id, target))
        return (target, None)

    # TODO(shu): added scenario type and stage type
    h5_output_file = '_{:05d}'.format(group_id)
    logging.info('writing {} messages to h5 file {} for target {}'
                 .format(grading_mtx.shape[0], h5_output_file, target))
    h5_utils.write_h5(grading_mtx, target, h5_output_file)

    GradingResults = namedtuple('grading_results',
                                ['relative_time',
                                 'speed',
                                 'acceleration_mean',
                                 'lateral_acceleration',
                                 'lateral_deceleration',
                                 'longitudinal_acceleration',
                                 'longitudinal_deceleration',
                                 'lat_acc_hit_bound',
                                 ])
    GradingArguments = namedtuple('grading_arguments',
                                  ['feature_name',
                                   'filter_name',
                                   'filter_mode',
                                   'filter_value',
                                   'weight'])
    GradingResults.__new__.__defaults__ = (None,) * len(GradingResults._fields)
    GradingArguments.__new__.__defaults__ = (None,) * len(GradingArguments._fields)

    profiling_conf = feature_utils.get_config_open_space_profiling()

    grading_group_result = GradingResults(
        acceleration_mean=compute_mean(
            grading_mtx,
            GradingArguments(
                feature_name='acceleration',
                weight=1.0
            ), profiling_conf.min_sample_size, FEATURE_IDX),
        lat_acc_hit_bound=compute_count(
            grading_mtx, GradingArguments(
                feature_name='lateral_acceleration_hit_bound',
            ), FEATURE_IDX),
    )
    return (target, grading_group_result)


def output_result(target_grading):
    """Write the grading results to files in corresponding target dirs"""
    target_dir, grading = target_grading
    grading_output_path = os.path.join(target_dir, 'open_space_performance_grading.txt')
    logging.info('writing grading output {} to {}'.format(grading, target_dir))

    grading_dict = grading._asdict()
    with open(grading_output_path, 'w') as grading_file:
        grading_file.write('Grading_output: \t {0:<36s} {1:<16s} {2:<16s} {3:<16s}\n'
                           .format('Grading Items', 'Grading Values', 'Sampling Size',
                                   'Event Timestamp'))
        for name, value in grading_dict.items():
            if not value:
                logging.warning('grading value for {} is None'.format(name))
                continue

            grading_file.write('Grading_output: \t {0:<36s} {1:<16.3%} {2:<16n} \n'
                               .format(name, value[0], value[1]))
