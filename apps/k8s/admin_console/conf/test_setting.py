#!/usr/bin/env python3
"""
Test environment configuration module
"""

from conf import base_setting


class TestingConfig(base_setting.Config):
    """
    Test environment configuration class
    """
    DB_HOST = ""
    TESTING = True
