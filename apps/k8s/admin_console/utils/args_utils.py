"""
Some tool functions from front end
"""

import flask


def get_args(*args):
    """
    Get the args from front end
    """
    get_dict = {}
    for key in args:
        if isinstance(key, tuple):
            value = flask.request.args.get(key[0], key[1])
            key = key[0]
        else:
            value = flask.request.args.get(key)
        if value and value.isdigit():
            value = int(value)
        get_dict[key] = value
    return get_dict
