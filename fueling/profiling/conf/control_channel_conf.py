#!/usr/bin/env python

"""Configs to store the necessary dict or list"""

MODE_IDX = {
    # Modes extracted from all vehicle types
    "driving_mode": 0,
    "gear_location": 1,
    "timestamp_sec": 2,
    "sequence_num": 3,
    # Additional Modes extracted from Lincoln vehicle type
    "throttle_chassis": 4,
    "brake_chassis": 5,
}

FEATURE_IDX = {
    # Features extracted from Control Channel
    "station_reference": 0,
    "speed_reference": 1,
    "acceleration_reference": 2,
    "heading_reference": 3,
    "heading_rate_reference": 4,
    "curvature_reference": 5,
    "path_remain": 6,
    "station_error": 7,
    "speed_error": 8,
    "lateral_error": 9,
    "lateral_error_rate": 10,
    "heading_error": 11,
    "heading_error_rate": 12,
    "throttle_cmd": 13,
    "brake_cmd": 14,
    "acceleration_cmd": 15,
    "steering_cmd": 16,
    "station": 17,
    "speed": 18,
    "acceleration": 19,
    "jerk": 20,
    "lateral_acceleration": 21,
    "lateral_jerk": 22,
    "heading_angle": 23,
    "heading_rate": 24,
    "heading_acceleration": 25,
    "heading_jerk": 26,
    "total_time": 27,
    "total_time_exceeded": 28,
    "timestamp_sec": 29,
    "sequence_num": 30,
    "localization_timestamp_sec": 31,
    "localization_sequence_num": 32,
    "chassis_timestamp_sec": 33,
    "chassis_sequence_num": 34,
    "trajectory_timestamp_sec": 35,
    "trajectory_sequence_num": 36,
    # Additional Features extracted from Chassis Channel
    "throttle_chassis": 37,
    "brake_chassis": 38,
    "pose_heading_offset": 39,
}

POSE_IDX = {
    # Features extracted from Localization Channel
    "timestamp_sec": 0,
    "sequence_num": 1,
    "pose_position_x": 2,
    "pose_position_y": 3,
    "pose_heading": 4,
}

FEATURE_NAMES = ["station_reference", "speed_reference", "acceleration_reference", "heading_reference", "heading_rate_reference",
                 "curvature_reference", "path_remain", "station_error", "speed_error", "lateral_error", "lateral_error_rate",
                 "heading_error", "heading_error_rate", "throttle_cmd", "brake_cmd", "acceleration_cmd", "steering_cmd", "station",
                 "speed", "acceleration", "jerk", "lateral_acceleration", "lateral_jerk", "heading_angle", "heading_rate",
                 "heading_acceleration", "heading_jerk", "total_time", "total_time_exceeded", "timestamp_sec", "sequence_num",
                 "localization_timestamp_sec", "localization_sequence_num", "chassis_timestamp_sec", "chassis_sequence_num",
                 "trajectory_timestamp_sec", "trajectory_sequence_num", "throttle_chassis", "brake_chassis", "pose_heading_offset"]
