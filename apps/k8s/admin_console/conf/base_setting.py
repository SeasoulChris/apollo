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

    # MongoDB config
    DB_HOST = ""
    DB_PORT = 27017
    DB_NAME = "fuel"
