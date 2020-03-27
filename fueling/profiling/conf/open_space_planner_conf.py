#!/usr/bin/env python

"""Configs to store the necessary dict or list"""

# TODO think of better way to merge these two configs
FEATURE_IDX = {
    'relative_time': 0,
    'speed': 1, # not sure if needed
    'acceleration': 2,
    'acceleration_ratio': 3,
    'deceleration_ratio': 4,
    'lateral_acceleration_ratio': 5,
}

FEATURE_NAMES = [
    'relative_time',
    'speed', # not sure if needed
    'acceleration',
    'acceleration_ratio',
    'deceleration_ratio',
    'lateral_acceleration_ratio',
]
