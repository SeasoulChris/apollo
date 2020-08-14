#!/usr/bin/env python3
"""
The logical control of the account in front of the view
"""

import datetime
import json

import flask

from common import paginator
from controllers import job
from fueling.common import account_utils
from utils import time_utils

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
        new_operation = account_data.get("new_operation")
        operations = account_data.get("operations")
        if start_time:
            account_data["apply_date"] = time_utils.get_datetime_str(account_data["apply_date"])
        if end_time:
            account_data["due_date"] = time_utils.get_datetime_str(account_data["due_date"])
        if operations:
            for opt in operations:
                opt["time"] = time_utils.get_datetime_str(opt["time"])
                try:
                    opt["action"]["due_date"] = time_utils.get_datetime_str(
                        opt["action"]["due_date"])
                except KeyError:
                    pass
        if new_operation:
            new_operation["time"] = time_utils.get_datetime_str(new_operation["time"])
            try:
                new_operation["action"]["due_date"] = (time_utils.get_datetime_str(
                    new_operation["action"]["due_date"]))
            except KeyError:
                pass
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
        obj["add_quota"] = True
    return objs


def extension_date(objs, days):
    """
    Modify the due_date for the obj
    """
    for obj in objs:
        due_date = obj.get("due_date")
        now_date = datetime.datetime.now()
        add_date = datetime.timedelta(days=days)
        if obj["status"] == "Enabled":
            if due_date:
                obj["due_date"] = due_date + add_date
            else:
                obj["due_date"] = now_date + add_date
        elif obj["status"] in ("Over-quota", "Expired"):
            obj["due_date"] = now_date + add_date
            obj["status"] = "Enabled"
        else:
            continue
        account_db.save_account_due_date(obj["_id"], obj["due_date"])
    return objs


def save_operations(objs):
    """
    Save the operations
    """
    for obj in objs:
        update_dict = {}
        email = flask.session.get("user_info").get("email")
        update_dict["email"] = email
        is_add_quota = obj.get("add_quota")
        if is_add_quota:
            remaining_quota = obj["remaining_quota"]
            due_date = obj["due_date"]
            update_dict["action"] = {"remaining_quota": remaining_quota, "due_date": due_date}
        else:
            update_dict["action"] = {}
        diff_dict = obj.get("diff_services")
        if diff_dict:
            update_dict["action"].update(diff_dict)
        update_dict = account_db.save_account_operation_msg(obj["_id"], update_dict)
        obj["new_operation"] = update_dict
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
        job_filters = [{"$match": {"vehicle_sn": obj["vehicle_sn"], "is_valid": True}},
                       {"$group": {"_id": "$job_type", "count": {"$sum": 1}}}]
        job_type_counts = job.job_collection.aggregate(job_filters)
        job_counts_dict = {}
        for job_type_count in job_type_counts:
            job_counts_dict[job_type_count["_id"]] = job_type_count["count"]
        if services:
            services.sort(key=lambda x: x["job_type"])
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
            due_date = obj.get("due_date")
            if due_date and is_expired(due_date):
                obj["status"] = "Expired"
    return objs


def update_services(objs, services_list):
    """Update the account services"""
    for obj in objs:
        diff_dict = compare_services(obj["services"], services_list)
        if diff_dict:
            account_db.save_account_services(obj["_id"], services_list)
            obj["services"] = services_list
            obj["diff_services"] = diff_dict
    return objs


def is_over_quota(used, quota):
    """Whether the account is over quota"""
    return used > quota


def is_expired(expire_date):
    """Whether the account is expired"""
    return expire_date < datetime.datetime.now()


def get_account_filter(conf_dict, args_dict):
    """
    Get the filter about account
    """
    black_list = conf_dict["black_list"]
    vehicle_sn = args_dict["vehicle_sn"]
    find_filter = []
    if vehicle_sn:
        find_filter.append({"vehicle_sn": vehicle_sn})
    else:
        for black_sn in black_list:
            find_filter.append({"vehicle_sn": {'$ne': black_sn}})
    if find_filter:
        filters = {"$and": find_filter}
    else:
        filters = {}
    return filters


def get_account_objs(filters):
    """
    Get the accounts from collection by filters
    """
    accounts = account_db.get_account_info(filters)
    account_used = get_job_used(accounts)
    account_objs = format_account_time(account_used)
    account_list = sorted(account_objs,
                          key=lambda x: x.get("apply_date"), reverse=True)
    return account_list


def get_account_paginator(current_page, nums, default_pages=10):
    """
    Get the current page obj and the content index
    """
    account_paginator = paginator.Pagination(nums, default_pages)
    current_page = paginator.CurrentPaginator(current_page, account_paginator)
    first, last = current_page.get_index_content()
    return current_page, (first, last)


def get_virtual_vehicle_sn(filename):
    """
    Get the virtual vehicle sn
    """
    prefix = "SK" + str(datetime.datetime.now().year)
    with open(filename, "r+") as f:
        conf_dict = json.load(f)
        vehicle_sn_num = conf_dict.get("vehicle_sn_suffix")
        vehicle_sn_suffix = str(vehicle_sn_num)
        zero_filler = ""
        len_vehicle_sn_suffix = len(vehicle_sn_suffix)
        while 3 - len_vehicle_sn_suffix > 0:
            zero_filler += "0"
            len_vehicle_sn_suffix += 1
        vehicle_sn_suffix = zero_filler + vehicle_sn_suffix
        conf_dict["vehicle_sn_suffix"] = vehicle_sn_num + 1
        f.seek(0, 0)
        f.truncate()
        json.dump(conf_dict, f)
    return prefix + vehicle_sn_suffix


def stamp_account_time(objs):
    """
    stamp the account time
    """
    for account_data in objs:
        start_time = account_data.get("apply_date")
        end_time = account_data.get("due_date")
        new_operation = account_data.get("new_operation")
        operations = account_data.get("operations")
        if start_time:
            account_data["apply_date"] = time_utils.\
                get_datetime_timestamp(account_data["apply_date"])
        if end_time:
            account_data["due_date"] = time_utils.get_datetime_timestamp(account_data["due_date"])
        if operations:
            for opt in operations:
                opt["time"] = time_utils.get_datetime_timestamp(opt["time"])
                opt["email"] = "administrator"
                try:
                    opt["action"]["due_date"] = time_utils.get_datetime_timestamp(
                        opt["action"]["due_date"])
                except KeyError:
                    pass
        if new_operation:
            new_operation["time"] = time_utils.get_datetime_timestamp(new_operation["time"])
            try:
                new_operation["action"]["due_date"] = (time_utils.time_utils.get_datetime_timestamp
                                                       (new_operation["action"]["due_date"]))
            except KeyError:
                pass
    return objs


def compare_services(old_services, new_services):
    """
    compare the list diff services
    """
    old_set = set()
    new_set = set()
    for service in old_services:
        old_set.add((service["job_type"], service["status"]))
    for service in new_services:
        new_set.add((service["job_type"], service["status"]))
    diff_set = new_set - old_set
    if diff_set:
        diff_dict = {"type": [], "status": []}
        for diff in diff_set:
            diff_dict["type"].append(diff[0])
            diff_dict["status"].append(diff[1])
    else:
        diff_dict = {}
    return diff_dict
