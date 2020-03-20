#!/usr/bin/env python

import numpy as np


def filter_value(grading_mtx, column_name, threshold, filter_mode=0):
    """Filter the rows out if they do not satisfy threshold values"""
    if filter_mode == 0:
        return np.delete(grading_mtx,
                         np.where(np.fabs(grading_mtx[:, column_name]) < threshold), axis=0)

    if filter_mode == 1:
        return np.delete(grading_mtx,
                         np.where(np.fabs(grading_mtx[:, column_name]) >= threshold), axis=0)
    return grading_mtx
