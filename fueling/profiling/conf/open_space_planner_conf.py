#!/usr/bin/env python

"""Configs to store the necessary dict or list"""


TRAJECTORY_FEATURE_NAMES = [
    'relative_time',
    'speed', # not sure if needed
    'acceleration',
    'acceleration_ratio',
    'deceleration_ratio',
    'longitudinal_acceleration_ratio',
    'longitudinal_deceleration_ratio',
    'lateral_acceleration_ratio',
    'lateral_deceleration_ratio',
]

# TODO: think of better way to merge these two configs
FEATURE_IDX = {
    # latency features
    'end_to_end_time': 0,
    'zigzag_time': 1,

    # zigzag feature
    'non_gear_switch_length_ratio': 0,
}

# adding TRAJECTORY_FEATURE idx into FEATURE_IDX
for (idx, name) in enumerate(TRAJECTORY_FEATURE_NAMES):
    FEATURE_IDX[name] = idx
