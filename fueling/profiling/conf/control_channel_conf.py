#!/usr/bin/env python

"""Configs to store the necessary dict or list"""

"""Control Profiling Index and Names: """

MODE_IDX = {
    # Modes extracted from all vehicle types
    "driving_mode": 0,
    "gear_location": 1,
    "timestamp_sec": 2,
    "sequence_num": 3,
    # Additional Modes extracted from Lincoln vehicle type
    "throttle_chassis": 4,
    "brake_chassis": 5,
    "steering_chassis": 6,
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
    "reference_position_x": 27,
    "reference_position_y": 28,
    "total_time": 29,
    "total_time_exceeded": 30,
    "timestamp_sec": 31,
    "sequence_num": 32,
    "localization_timestamp_sec": 33,
    "localization_sequence_num": 34,
    "chassis_timestamp_sec": 35,
    "chassis_sequence_num": 36,
    "trajectory_timestamp_sec": 37,
    "trajectory_sequence_num": 38,
    # Additional Features extracted from Chassis Channel
    "throttle_chassis": 39,
    "brake_chassis": 40,
    "steering_chassis": 41,
    "pose_position_x": 42,
    "pose_position_y": 43,
    "pose_heading": 44,
    "pose_heading_offset": 45,
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
                 "heading_acceleration", "heading_jerk", "reference_position_x", "reference_position_y", "total_time",
                 "total_time_exceeded", "timestamp_sec", "sequence_num", "localization_timestamp_sec", "localization_sequence_num",
                 "chassis_timestamp_sec", "chassis_sequence_num", "trajectory_timestamp_sec", "trajectory_sequence_num",
                 "throttle_chassis", "brake_chassis", "steering_chassis", "pose_position_x", "pose_position_y", "pose_heading",
                 "pose_heading_offset"]

"""Vehicle Dynamics Profiling Index and Names: """


DYNAMICS_MODE_IDX = {
    # Modes extracted from all vehicle types
    "driving_mode": 0,
    "gear_location": 1,
    "timestamp_sec": 2,
    "sequence_num": 3,
    # Additional Modes extracted from Lincoln vehicle type
    "throttle_chassis": 4,
    "brake_chassis": 5,
}

DYNAMICS_FEATURE_IDX = {
    "throttle_cmd": 0,
    "brake_cmd": 1,
    "acceleration_cmd": 2,
    "steering_cmd": 3,
    "acceleration": 4,
    "steering": 5,
    "timestamp_sec": 6,
    "sequence_num": 7,
    "chassis_timestamp_sec": 8,
    "chassis_sequence_num": 9,
    "throttle": 10,
    "brake": 11,
}

DYNAMICS_FEATURE_NAMES = ["throttle_cmd", "brake_cmd", "acceleration_cmd", "steering_cmd", "acceleration", "streeing",
                          "timestamp_sec", "sequence_num", "chassis_timestamp_sec", "chassis_sequence_num",
                          "throttle", "brake"]
