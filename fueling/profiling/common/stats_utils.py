#!/usr/bin/env python

import numpy as np

import fueling.common.logging as logging


def compute_std(grading_mtx, arg, min_sample_size, FEATURE_IDX):
    """Compute the standard deviation"""
    grading_mtx = apply_filter(grading_mtx, arg, FEATURE_IDX)
    if grading_mtx is None:
        return (0.0, 0)

    elem_num, _ = grading_mtx.shape
    if not check_data_size(elem_num, min_sample_size):
        return (0.0, 0)

    column_norm = grading_mtx[:, FEATURE_IDX[arg.std_norm_name]]
    column_denorm = grading_mtx[:, np.array([FEATURE_IDX[denorm_name]
                                             for denorm_name in arg.std_denorm_name])]
    column_denorm = np.maximum(np.fabs(column_denorm), arg.std_max_compare)
    column_denorm = [np.prod(column) for column in column_denorm]
    std = [nor / (denor * arg.std_denorm_weight)
           for nor, denor in zip(column_norm, column_denorm)]
    return (get_std_value(std), elem_num)


def compute_peak(grading_mtx, arg, min_sample_size, FEATURE_IDX):
    """Compute the peak value"""
    grading_mtx = apply_filter(grading_mtx, arg, FEATURE_IDX)
    if grading_mtx is None:
        return ([0.0, 0.0], 0)

    elem_num, _ = grading_mtx.shape
    if not check_data_size(elem_num, min_sample_size):
        return ([0.0, 0.0], 0)

    idx_max = np.argmax(
        np.fabs(grading_mtx[:, FEATURE_IDX[arg.feature_name]]))
    return ([np.fabs(grading_mtx[idx_max, FEATURE_IDX[arg.feature_name]]) /
             arg.threshold, grading_mtx[idx_max, FEATURE_IDX[arg.time_name]]],
            elem_num)


def compute_ending(grading_mtx, arg, FEATURE_IDX):
    """Compute the specific value at the final state"""
    grading_mtx = apply_filter(grading_mtx, arg, FEATURE_IDX)
    if grading_mtx is None:
        return ([[0.0], [0.0], [0.0]], 0)

    elem_num, _ = grading_mtx.shape
    if not check_data_size(elem_num, 2):
        return ([[0.0], [0.0], [0.0]], 0)

    grading_mtx = grading_mtx[np.argsort(grading_mtx[:, FEATURE_IDX[arg.time_name]])]
    static_error = [np.fabs(grading_mtx[0, FEATURE_IDX[arg.feature_name]]) / arg.threshold]
    static_start_time = [grading_mtx[0, FEATURE_IDX[arg.time_name]]]
    static_stop_time = [grading_mtx[0, FEATURE_IDX[arg.time_name]]]
    for idx in range(1, grading_mtx.shape[0]):
        if (grading_mtx[idx, FEATURE_IDX[arg.time_name]] -
                grading_mtx[idx - 1, FEATURE_IDX[arg.time_name]] <= 1.0):
            static_stop_time[-1] = grading_mtx[idx, FEATURE_IDX[arg.time_name]]
        else:
            static_error.append(np.fabs(grading_mtx[idx, FEATURE_IDX[arg.feature_name]]) /
                                arg.threshold)
            static_start_time.append(grading_mtx[idx, FEATURE_IDX[arg.time_name]])
            static_stop_time.append(grading_mtx[idx, FEATURE_IDX[arg.time_name]])
    return ([static_error, static_start_time, static_stop_time], elem_num)


def compute_usage(grading_mtx, arg, min_sample_size, FEATURE_IDX):
    """Compute the usage value"""
    grading_mtx = apply_filter(grading_mtx, arg, FEATURE_IDX)
    if grading_mtx is None:
        return (0.0, 0)

    elem_num, _ = grading_mtx.shape
    if not check_data_size(elem_num, min_sample_size):
        return (0.0, 0)
    return (get_std_value([val / arg.weight
                           for val in grading_mtx[:, FEATURE_IDX[arg.feature_name]]]),
            elem_num)


def compute_beyond(grading_mtx, arg, FEATURE_IDX):
    """Compute the beyond_the_threshold counting value"""
    elem_num, _ = grading_mtx.shape
    return (len(np.where(np.fabs(grading_mtx[:, FEATURE_IDX[arg.feature_name]]) >=
                         arg.threshold)[0]) / elem_num,
            elem_num)


def compute_count(grading_mtx, arg, FEATURE_IDX):
    """Compute the event (boolean true) counting value"""
    elem_num, _ = grading_mtx.shape
    return (len(np.where(grading_mtx[:, FEATURE_IDX[arg.feature_name]] == 1)[0]) / elem_num,
            elem_num)


def compute_mean(grading_mtx, arg, min_sample_size, FEATURE_IDX):
    """Compute the mean value"""
    grading_mtx = apply_filter(grading_mtx, arg, FEATURE_IDX)
    if grading_mtx is None:
        return (0.0, 0)

    elem_num, item_num = grading_mtx.shape
    if not check_data_size(elem_num, min_sample_size):
        return (0.0, 0)

    if item_num <= FEATURE_IDX[arg.feature_name]:
        logging.warning(F'No desired feature item for mean computing: required '
                        F'{FEATURE_IDX[arg.feature_name]}, but found {item_num}')
        return (0.0, 0)
    return (np.mean(grading_mtx[:, FEATURE_IDX[arg.feature_name]], axis=0) / arg.weight,
            elem_num)


def apply_filter(grading_mtx, arg, FEATURE_IDX):
    if arg.filter_name:
        if (len(arg.filter_name) != len(arg.filter_value)
            or len(arg.filter_name) != len(arg.filter_mode)):
            logging.warning('Number of filter attributes are not equal!')
            return None

        for idx, name in enumerate(arg.filter_name):
            grading_mtx = filter_value(grading_mtx, FEATURE_IDX[name],
                                       arg.filter_value[idx], arg.filter_mode[idx])
    return grading_mtx


def filter_value(grading_mtx, column_name, threshold, filter_mode=0):
    """Filter the rows out if they do not satisfy threshold values"""
    if filter_mode == 0:
        return np.delete(grading_mtx,
                         np.where(np.fabs(grading_mtx[:, column_name]) < threshold), axis=0)
    if filter_mode == 1:
        return np.delete(grading_mtx,
                         np.where(np.fabs(grading_mtx[:, column_name]) >= threshold), axis=0)
    return grading_mtx


def check_data_size(elem_num, min_sample_size):
    if elem_num < min_sample_size:
        logging.warning(F'Not enough elements for computing: {elem_num} < {min_sample_size}')
        return False
    return True


def get_std_value(grading_column):
    """Calculate the standard deviation value"""
    return (sum(val**2 for val in grading_column) / (len(grading_column) - 1)) ** 0.5
