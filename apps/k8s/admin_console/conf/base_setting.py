#!/usr/bin/env python3
"""
Basic configuration module
"""

import os


class Config(object):
    """
    Basic configuration class
    """
    # Flask config
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.urandom(24)

    # Job config
    JOB_TYPE = {"A": "All", "VC": "vehicle_calibration",
                "SC": "sensor_calibration", "CP": "virtual_lane_generation"}
    SHOW_JOB_TYPE = {"A": "所有", "VC": "Vehicle Calibration", "SC": "Sensor Calibration",
                     "CP": "Virtual Lane Generation"}
    TIME_FIELD = {"All": 0, "7d": 7, "30d": 30, "1y": 365}
    SHOW_TIME_FIELD = {"All": "所有", "7d": "过去7天", "30d": "过去30天", "1y": "1年前"}
    BLACK_LIST = ["CH0000000"]
