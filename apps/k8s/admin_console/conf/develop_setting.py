#!/usr/bin/env python3
"""
Develop environment configuration module
"""

from conf import base_setting


class DevelopConfig(base_setting.Config):
    """
    Develop environment configuration class
    """
    DB_HOST = "192.168.141.20"
    DEBUG = True
