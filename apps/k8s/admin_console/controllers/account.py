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
    jobs = []
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
        jobs.append(account_data)
    return jobs


def get_job_used(objs):
    """
    statistics used
    """
    for obj in objs:
        services = obj.get("services")
        job_filters = [{"$match": {"vehicle_sn": obj["vehicle_sn"]}},
                       {"$group": {"_id": "$job_type", "count": {"$sum": 1}}}]
        logging.info(f"services:{services}")
        counts = job.job_collection.aggregate(job_filters)
        for c in counts:
            logging.info(f"count:{c}")
        if services:
            sum_counts = 0
            logging.info(f"name:{obj['com_email']}")
            for count in counts:
                for service in services:
                    if service["job_type"] == count["_id"]:
                        logging.info(f"count:{count}")
                        logging.info(f"service:{service}")
                        job_count = count["count"]
                        service["used"] = job_count
                        sum_counts += job_count
            obj["remaining_quota"] = obj["quota"] - sum_counts
            logging.info(f"quota:{obj['quota']}")
            logging.info(f"sum_counts:{sum_counts}")

            if is_over_quota(sum_counts, obj.get("quota")) and obj["status"] == "Enabled":
                obj["status"] = "Over-quota"
            if is_expired(obj["due_date"]):
                obj["status"] = "Expired"
    return objs


def is_over_quota(used, quota):
    return used > quota


def is_expired(expire_date):
    return expire_date < datetime.datetime.now()
