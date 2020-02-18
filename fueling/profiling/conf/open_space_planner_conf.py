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
}

FEATURE_NAMES = [
    'relative_time',
    'speed',
    'acceleration',
]
