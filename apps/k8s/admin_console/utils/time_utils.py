"""
Some tool functions about time
"""

import datetime


def get_datetime_str(date_obj):
    """
    Converts the time object to a string
    """
    if isinstance(date_obj, datetime.datetime):
        return date_obj.strftime("%Y-%m-%d %H:%m:%S")
    return date_obj


def days_ago(day):
    """
    Get the past timestamp
    """
    return datetime.datetime.now() - datetime.timedelta(days=day)
