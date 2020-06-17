#!/usr/bin/env python3
"""
Production environment configuration module
"""

from conf import base_setting


class ProductionConfig(base_setting.Config):
    """
    Production environment configuration class
    """
    DB_HOST = ""
