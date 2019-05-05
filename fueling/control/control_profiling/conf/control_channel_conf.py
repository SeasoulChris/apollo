#!/usr/bin/env python

"""Configs to store the necessary dict or list"""

MODE_IDX = {
    # Modes extracted from all vehicle types
    "driving_mode": 0,
    "gear_location": 1,
    "timestamp_sec": 2,
    # Additional Modes extracted from Lincoln vehicle type
    "throttle_chassis": 3,
    "brake_chassis": 4,
}

FEATURE_IDX= {
    # Features extracted from Control Channel
    "station_reference": 0,
    "speed_reference": 1,
    "acceleration_reference": 2,
    "heading_reference": 3,
    "heading_rate_reference": 4,
    "curvature_reference": 5,
    "station_error": 6,
    "speed_error": 7,
    "lateral_error": 8,
    "lateral_error_rate": 9,
    "heading_error": 10,
    "heading_error_rate": 11,
    "throttle_cmd": 12,
    "brake_cmd": 13,
    "acceleration_cmd": 14,
    "steering_cmd": 15,
    "station": 16,
    "speed": 17,
    "acceleration": 18,
    "jerk": 19,
    "lateral_acceleration": 20,
    "lateral_jerk": 21,
    "heading_angle": 22,
    "heading_rate": 23,
    "heading_acceleration": 24,
    "heading_jerk": 25,
    "total_time": 26,
    "total_time_exceeded": 27,
    "timestamp_sec": 28,
    # Additional Features extracted from Chassis Channel
    "throttle_chassis": 29,
    "brake_chassis": 30,
}

FEATURE_NAMES = ["station_reference","speed_reference","acceleration_reference","heading_reference","heading_rate_reference",
                 "curvature_reference", "station_error", "speed_error", "lateral_error", "lateral_error_rate", "heading_error",
                 "heading_error_rate", "throttle_cmd", "brake_cmd", "acceleration_cmd", "steering_cmd", "station", "speed",
                 "acceleration", "jerk", "lateral_acceleration", "lateral_jerk", "heading_angle", "heading_rate",
                 "heading_acceleration", "heading_jerk", "total_time", "total_time_exceeded", "timestamp_sec", "throttle_chassis",
                 "brake_chassis"]
