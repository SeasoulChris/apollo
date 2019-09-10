#!/usr/bin/env python
"""Time related utils."""

from datetime import datetime, timedelta

import pytz


def msg_time_to_datetime(msg_time, local_tz='America/Los_Angeles', msg_tz='UTC'):
    """Cyber message timestamp is generally in UTC nano seconds."""
    dt = datetime.fromtimestamp(msg_time / (10 ** 9), pytz.timezone(msg_tz))
    return dt.astimezone(pytz.timezone(local_tz))


def n_days_ago(n, fmt=None):
    """Get datetime of n days ago, return string if format is provided."""
    dt = datetime.now() - timedelta(days=n)
    return dt.strftime(fmt) if fmt else dt
