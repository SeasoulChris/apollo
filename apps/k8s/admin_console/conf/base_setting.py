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

    # UUAP authentication configuration
    UUAP_HOST = 'https://itebeta.baidu.com'
    UUAP_POST = '443'
    UUAP_P_TOKEN = 'UUAP_P_TOKEN_OFFLINE'
    UUAP_S_TOKEN = 'UUAP_S_TOKEN'
    UUAP_APP_KEY = 'uuapclient-477979544813760513-31dHl'
    UUAP_SECRET_KEY = '5290c0f97409407798def6'
    TOKEN_TIMEOUT = 60
    SERVER_NAME = "usa-data.baidu.com:8000"
