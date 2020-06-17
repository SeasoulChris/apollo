#!/usr/bin/env python3
"""
Basic configuration module
"""


class Config(object):
    """
    Basic configuration class
    """
    # Flask config
    DEBUG = False
    TESTING = False

    # MongoDB config
    DB_HOST = ""
    DB_PORT = 27017
    DB_NAME = "fuel"
