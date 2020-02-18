#!/usr/bin/env python

""" Open-space planner feature processing related utils. """

from collections import namedtuple
import numpy as np

import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging

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

    grading_results = namedtuple('grading_results',
                                 ['speed',
                                  'acceleration',
                                  ])
    grading_arguments = namedtuple('grading_arguments',
                                   ['mean_feature_name',
                                    'mean_filter_name',
                                    'mean_filter_mode',
                                    'mean_weight'])
    grading_results.__new__.__defaults__ = (
        None,) * len(grading_results._fields)
    grading_arguments.__new__.__defaults__ = (
        None,) * len(grading_arguments._fields)
    grading_group_result = grading_results(
        acceleration=compute_mean(grading_mtx, grading_arguments(
            mean_feature_name='acceleration',
            mean_weight=1.0
        )),
    )
    return (target, grading_group_result)


def compute_mean(grading_mtx, arg):
    """Compute the mean value"""
    profiling_conf = feature_utils.get_config_open_space_profiling()
    if arg.mean_filter_name:
        for idx in range(len(arg.mean_filter_name)):
            grading_mtx = filter_value(grading_mtx, FEATURE_IDX[arg.mean_filter_name[idx]],
                                       arg.mean_filter_value[idx], arg.mean_filter_mode[idx])
    elem_num, item_num = grading_mtx.shape
    if elem_num < profiling_conf.min_sample_size:
        logging.warning('no enough elements {} for mean computing requirement {}'
                        .format(elem_num, profiling_conf.min_sample_size))
        return (0.0, 0)
    if item_num <= FEATURE_IDX[arg.mean_feature_name]:
        logging.warning('no desired feature item {} for mean computing requirement {}'
                        .format(item_num, FEATURE_IDX[arg.mean_feature_name]))
        return (0.0, 0)
    return (np.mean(grading_mtx[:, FEATURE_IDX[arg.mean_feature_name]], axis=0) / arg.mean_weight,
            elem_num)
