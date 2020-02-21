#!/usr/bin/env python

"""Configs to store the necessary dict or list"""

"""Open Space Planner Profiling Index and Names: """
# FEATURE_IDX = {
#     'timestamp_sec': 0,
#     'sequency_num': 1,
#     # Features extracted from Planning Channel
#     'time_latency': 2,
#     'acceleration': 3,
# }

FEATURE_IDX = {
    'relative_time': 0,
    'speed': 1,
    'acceleration': 2,
    'lateral_acceleration': 3,
    'lateral_deceleration': 4,
    'longitudinal_acceleration': 5,
    'longitudinal_deceleration': 6,
    'lateral_acceleration_hit_bound': 7
}

FEATURE_NAMES = [
    'relative_time',
    'speed',
    'acceleration',
    'lateral_acceleration',
    'lateral_deceleration',
    'longitudinal_acceleration',
    'longitudinal_deceleration',
    'lateral_acceleration_hit_bound',
]
