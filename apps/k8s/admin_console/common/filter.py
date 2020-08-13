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


def get_en_action(is_valid):
    """
    Get english action
    """
    return "Invalid" if is_valid else "Valid"


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
    """
    Get failure cause of job
    """
    failure_cause = application.app.config.get("FAILURE_CAUSE")
    return failure_cause[code] if code and failure_cause.get("code") else code


def truncation_job_id(job_id):
    """
    Truncation the job id
    """
    return job_id[-6:] if job_id else ""


def get_account_show_status(status):
    """
    Get the account show status
    """
    status_dict = application.app.config.get("ACCOUNT_STATUS_FIELD")
    show_status = status_dict.get(status)
    return show_status if show_status else status


def get_account_show_region(region):
    """
    Get the account show region
    """
    region_dict = application.app.config.get("ACCOUNT_REGION_FIELD")
    show_region = region_dict.get(region)
    return show_region if show_region else region


def get_account_show_action(action):
    """
    Get the account action
    """
    action_dict = application.app.config.get("ACCOUNT_ACTION_FIELD")
    show_action = action_dict.get(action)
    return show_action if show_action else action
