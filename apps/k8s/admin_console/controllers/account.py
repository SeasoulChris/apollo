#!/usr/bin/env python3
"""
The logical control of the account in front of the view
"""

import datetime

from controllers import job
from fueling.common import account_utils
from utils import time_utils
from fueling.common import logging

account_db = account_utils.AccountUtils()


def get_account_list():
    """
    Get account list
    """
    accounts = []
    for account in account_db.db.find():
        account['_id'] = account['_id'].__str__()
        accounts.append(account)
    return accounts


def format_account_time(objs):
    """
    Format the job timestamp into a string
    """
    for account_data in objs:
        start_time = account_data.get("apply_date")
        end_time = account_data.get("due_date")
        operations = account_data.get("operations")
        if start_time:
            account_data["apply_date"] = time_utils.get_datetime_str(account_data["apply_date"])
        if end_time:
            account_data["due_date"] = time_utils.get_datetime_str(account_data["due_date"])
        if operations:
            for opt in operations:
                opt["time"] = time_utils.get_datetime_str(opt["time"])
    return objs


def add_quota(objs, add_quota):
    """
    Add quota for the obj
    """
    for obj in objs:
        quota = obj["quota"]
        if obj["status"] == "Enabled":
            quota += add_quota
        elif obj["status"] in ("Over-quota", "Expired"):
            quota = obj["used"] + add_quota
        account_db.save_account_quota(obj["_id"], quota)
        obj["quota"] = quota
        obj["remaining_quota"] = quota - obj["used"]
    return objs


def extension_date(objs, days):
    """
    Modify the due_date for the obj
    """
    for obj in objs:
        if obj["status"] == "Enabled":
            obj["due_date"] = obj["due_date"] + datetime.timedelta(days=days)
        elif obj["status"] in ("Over-quota", "Expired"):
            obj["due_date"] = datetime.datetime.now() + datetime.timedelta(days=days)
            obj["status"] = "Enabled"
        account_db.save_account_due_date(obj["_id"], obj["due_date"])
    return objs


def get_show_action(show_action, objs):
    """
    Get the show_action
    """
    for obj in objs:
        obj["show_action"] = show_action[obj["status"]]
    return objs


def get_job_used(objs):
    """
    statistics used
    """
    for obj in objs:
        services = obj.get("services")
        job_filters = [{"$match": {"vehicle_sn": obj["vehicle_sn"]}},
                       {"$group": {"_id": "$job_type", "count": {"$sum": 1}}}]
        job_type_counts = job.job_collection.aggregate(job_filters)
        job_counts_dict = {}
        for job_type_count in job_type_counts:
            job_counts_dict[job_type_count["_id"]] = job_type_count["count"]

        if services:
            sum_counts = 0
            for service in services:
                job_type_used = job_counts_dict.get(service["job_type"])
                if job_type_used:
                    service["used"] = job_type_used
                    sum_counts += job_type_used
                else:
                    service["used"] = 0
            obj["used"] = sum_counts
            obj["remaining_quota"] = obj["quota"] - sum_counts
            if is_over_quota(sum_counts, obj.get("quota")) and obj["status"] == "Enabled":
                obj["status"] = "Over-quota"
            if is_expired(obj["due_date"]):
                obj["status"] = "Expired"
    return objs


def is_over_quota(used, quota):
    """Whether the account is over quota"""
    return used > quota


def is_expired(expire_date):
    """Whether the account is expired"""
    return expire_date < datetime.datetime.now()
