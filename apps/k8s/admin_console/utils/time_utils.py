"""
Some tool functions about time
"""

import datetime


def get_datetime_str(date_obj):
    """
    Converts the time object to a string
    """
    if isinstance(date_obj, datetime.datetime):
        cn_time = date_obj.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
        return cn_time.strftime("%Y-%m-%d %H:%M:%S")
    return date_obj


def days_ago(day):
    """
    Get the past timestamp
    """
    return datetime.datetime.now() - datetime.timedelta(days=day)
