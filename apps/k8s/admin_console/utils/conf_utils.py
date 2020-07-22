"""
Some tool functions from conf
"""

import application


def get_conf(*args):
    """
    Get the conf dict from args
    """
    get_dict = {}
    for key in args:
        get_dict[key.lower()] = application.app.config.get(key)
    return get_dict
