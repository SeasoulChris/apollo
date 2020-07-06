#!/usr/bin/env python3
"""
about filter function
"""

import application


def get_show_job_type(job):
    """
    Format job type use blank
    """
    return " ".join(map(lambda x: x.capitalize(), job.lower().split("_"))) if job else ""


def get_action(is_valid):
    """
    Get action by is_valid
    """
    return "设置无效" if is_valid else "设置有效"


def get_cn_action(en_action):
    """
    Get chinese action
    """
    if en_action.lower() == "valid":
        return "有效"
    elif en_action.lower() == "invalid":
        return "无效"
    else:
        return "错误"


def get_duration(time_tuple):
    """
    Time gap formatting
    """
    re = ""
    if time_tuple[0]:
        re += str(time_tuple[0]) + "天"
    hours, minutes, seconds = 0, 0, 0
    if time_tuple[1]:
        hours, seconds = divmod(time_tuple[1], 3600)
    if hours:
        re += str(hours) + "小时"
    if seconds:
        minutes, seconds = divmod(seconds, 60)
    if minutes:
        re += str(minutes) + "分钟"
    if seconds:
        re += str(seconds) + "秒"
    return re


def get_failure_cause(code):
    failure_cause = application.app.config.get("FAILURE_CAUSE")
    return failure_cause[code] if code else ""


def truncation_job_id(job_id):
    return job_id[-6:] if job_id else ""
