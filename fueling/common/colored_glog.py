#!/usr/bin/env python
"""
Glog with colors.

Usage:
    import fueling.common.colored_glog as glog

All other code should be the same as using glog.
"""

import glog
import termcolor


def _colored_debug(msg):
    return termcolor.colored(msg, 'grey') if msg else None


def _colored_info(msg):
    return termcolor.colored(msg, 'blue') if msg else None


def _colored_warn(msg):
    return termcolor.colored(msg, 'yellow') if msg else None


def _colored_error(msg):
    return termcolor.colored(msg, 'red') if msg else None


def _colored_fatal(msg):
    return termcolor.colored(msg, 'green', 'on_red') if msg else None


def _colored_check(msg):
    return _colored_fatal(msg)


def debug(msg, *args, **kwargs):
    """Equivalent to glog.debug()."""
    glog.debug(_colored_debug(msg), *args, **kwargs)


def info(msg, *args, **kwargs):
    """Equivalent to glog.info()."""
    glog.info(_colored_info(msg), *args, **kwargs)


def warn(msg, *args, **kwargs):
    """Equivalent to glog.warn()."""
    glog.warn(_colored_warn(msg), *args, **kwargs)


def error(msg, *args, **kwargs):
    """Equivalent to glog.error()."""
    glog.error(_colored_error(msg), *args, **kwargs)


def fatal(msg, *args, **kwargs):
    """Equivalent to glog.fatal()."""
    glog.fatal(_colored_fatal(msg), *args, **kwargs)


def check(condition, message=None):
    """Equivalent to glog.check()."""
    glog.check(condition, _colored_check(message))


def check_notnone(obj, message=None):
    """Equivalent to glog.check_notnone()."""
    glog.check_notnone(obj, _colored_check(message))


def check_eq(obj1, obj2, message=None):
    """Equivalent to glog.check_eq()."""
    glog.check_eq(obj1, obj2, _colored_check(message))


def check_ne(obj1, obj2, message=None):
    """Equivalent to glog.check_ne()."""
    glog.check_ne(obj1, obj2, _colored_check(message))


def check_ge(obj1, obj2, message=None):
    """Equivalent to glog.check_ge()."""
    glog.check_ge(obj1, obj2, _colored_check(message))


def check_gt(obj1, obj2, message=None):
    """Equivalent to glog.check_gt()."""
    glog.check_gt(obj1, obj2, _colored_check(message))


def check_le(obj1, obj2, message=None):
    """Equivalent to glog.check_le()."""
    glog.check_le(obj1, obj2, _colored_check(message))


def check_lt(obj1, obj2, message=None):
    """Equivalent to glog.check_lt()."""
    glog.check_lt(obj1, obj2, _colored_check(message))
