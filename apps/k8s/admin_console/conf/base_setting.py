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
    JOB_TYPE = ["All", "Vehicle Calibration", "Sensor Calibration", "Control Profiling"]
    TIME_FIELD = {"All": 0, "7 days before": 7, "30 days before": 30, "A year ago": 365}
