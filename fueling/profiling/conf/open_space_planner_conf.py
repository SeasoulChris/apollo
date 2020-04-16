#!/usr/bin/env python

"""Configs to store the necessary dict or list"""
TRAJECTORY_FEATURE_NAMES = [
    # from extract_data_from_trajectory_point()
    'relative_time',
    'kappa',
    'curvature_ratio',
    'speed',  # not sure if needed
    'acceleration',
    'longitudinal_acceleration',  # include both +/-
    'lateral_acceleration',  # include both +/-
    'acceleration_ratio',
    'deceleration_ratio',
    'longitudinal_acceleration_ratio',
    'longitudinal_deceleration_ratio',
    'lateral_acceleration_ratio',
    'lateral_deceleration_ratio',
    'distance_to_roi_boundaries_ratio',
    'distance_to_obstacles_ratio',
    'min_time_to_collision_ratio',

    # from calculate_jerk_ratios()
    'longitudinal_positive_jerk_ratio',
    'longitudinal_negative_jerk_ratio',
    'lateral_positive_jerk_ratio',
    'lateral_negative_jerk_ratio',

    # from calculate_dkappa_ratio()
    'curvature_change_ratio',
]

# TODO: think of better way to merge these two configs
FEATURE_IDX = {
    # planning stage feature
    'stage_completion_time': 0,
    'initial_heading_diff_ratio': 1,

    # latency features
    'end_to_end_time': 0,
    'zigzag_time': 1,

    # zigzag feature
    'non_gear_switch_length_ratio': 0,
}

# adding TRAJECTORY_FEATURE idx into FEATURE_IDX
for (idx, name) in enumerate(TRAJECTORY_FEATURE_NAMES):
    FEATURE_IDX[name] = idx

REFERENCE_VALUES = {
    'control_precision': 0.25,
    'distance_to_roi_boundary_buffer': 0.1,  # meter
    'time_to_collision': 6.0,  # second
    'max_time_to_collision': 10.0,  # second

    'lateral_jerk_positive_upper_bound': 1.0,
    'lateral_jerk_negative_upper_bound': -1.0,
    'longitudinal_jerk_positive_upper_bound': 2.0,
    'longitudinal_jerk_negative_upper_bound': -2.0,
}

REFERENCE_VALUES['roi_reference_distance'] = REFERENCE_VALUES['distance_to_roi_boundary_buffer'] + \
    REFERENCE_VALUES['control_precision'] / 2.0

REFERENCE_VALUES['obstacle_reference_distance'] = 1.7 * \
    REFERENCE_VALUES['control_precision']
