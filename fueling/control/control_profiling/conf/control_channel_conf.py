""" Configs to store the necessary dict or list """
#!/usr/bin/env python

FEATURE_INDEX = {
    "station_reference": 0,
    "speed_reference": 1,
    "acceleration_reference": 2,
    "heading_reference": 3,
    "curvature_reference": 4,
    "station_error": 5,
    "speed_error": 6,
    "lateral_error": 7,
    "lateral_error_rate": 8,
    "heading_error": 9,
    "heading_error_rate": 10,
    "throttle_cmd": 11,
    "brake_cmd": 12,
    "acceleration_cmd": 13,
    "steering_cmd": 14,
    "linear_velocity": 15,
    "heading angle": 16,
}

FEATURE_NAMES = ["station_reference", "speed_reference", "acceleration_reference",
                 "heading_reference", "curvature_reference", "station_error",
                 "speed_error", "lateral_error", "lateral_error_rate",
                 "heading_error", "heading_error_rate", "throttle_cmd", "brake_cmd", 
                 "acceleration_cmd", "steering_cmd", "linear_velocity", "heading angle"]
