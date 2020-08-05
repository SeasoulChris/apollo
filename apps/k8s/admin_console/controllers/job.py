#!/usr/bin/env python3
"""
The logical control of the Job in front of the view
"""

from common import paginator
from fueling.common import mongo_utils
from utils import time_utils

job_collection = mongo_utils.Mongo().fuel_job_collection()


def format_job_time(objs):
    """
    Format the job timestamp into a string
    """
    jobs = []
    for job_data in objs:
        job_data['_id'] = job_data['_id'].__str__()
        start_time = job_data.get("start_time")
        end_time = job_data.get("end_time")
        operations = job_data.get("operations")
        if start_time:
            job_data["start_time"] = time_utils.get_datetime_str(job_data["start_time"])
        if end_time:
            job_data["end_time"] = time_utils.get_datetime_str(job_data["end_time"])
        job_data["duration_time"] = (0, 0)
        if start_time and end_time:
            duration_time = end_time - start_time
            job_data["duration_time"] = (duration_time.days, duration_time.seconds)
        if operations:
            for opt in operations:
                opt["time"] = time_utils.get_datetime_str(opt["time"])
        jobs.append(job_data)
    return jobs


def format_job_for_services(objs):
    """
    Format the job data
    """
    jobs = []
    for job_data in objs:
        job_data['_id'] = job_data['_id'].__str__()
        start_time = job_data.get("start_time")
        end_time = job_data.get("end_time")
        operations = job_data.get("operations")
        failure_code = job_data.get("failure_code")
        failure_detail = job_data.get("failure_detail")
        failure_dic = {}
        if start_time:
            job_data["start_time"] = time_utils.get_datetime_timestamp(job_data["start_time"])
        if end_time:
            job_data["end_time"] = time_utils.get_datetime_timestamp(job_data["end_time"])
        job_data["duration"] = 0
        if start_time and end_time:
            duration_time = end_time - start_time
            job_data["duration"] = duration_time.seconds
        if operations:
            for opt in operations:
                opt["email"] = "Administrator"
                opt["time"] = time_utils.get_datetime_timestamp(opt["time"])
        if failure_detail:
            failure_dic["detail"] = failure_detail
            job_data.pop("failure_detail")
        if failure_code:
            failure_dic["code"] = failure_code
            job_data.pop("failure_code")
            job_data["failure"] = failure_dic

        jobs.append(job_data)
    return jobs


def get_jobs_count():
    """
    Get the total numbers of jobs
    """
    return job_collection.count()


def get_job_by_id(job_id):
    """
    Get the job obj by job_id
    """
    obj = job_collection.find_one({"job_id": job_id})
    return obj


def get_job_filter(conf_dict, args_dict):
    """
    Get the filter about job
    """
    res = {}
    filter_list = []
    job_selected = args_dict["job_selected"]
    job_type = conf_dict["job_type"]
    time_selected = args_dict["time_selected"]
    time_field = conf_dict["time_field"]
    vehicle_sn = args_dict["vehicle_sn"]
    black_list = conf_dict["black_list"]
    filter_list.append({"vehicle_sn": {"$exists": True}})
    filter_job_type = list(conf_dict["job_type"].values())
    filter_job_type.remove("All")
    filter_list.append({"job_type": {"$in": filter_job_type}})

    if job_selected == "A":
        pass
    elif job_selected in job_type:
        filter_list.append({"job_type": job_type[job_selected]})
    else:
        res["code"] = 300
        res["msg"] = "Invalid Parameter"

    if time_selected == "All":
        pass
    elif time_selected in time_field:
        days_ago = time_utils.days_ago(time_field[time_selected])
        filter_list.append({"start_time": {"$gt": days_ago}})
    else:
        res["code"] = 300
        res["msg"] = "Invalid Parameter"
    if vehicle_sn:
        filter_list.append({"vehicle_sn": vehicle_sn})
    else:
        for black_sn in black_list:
            filter_list.append({"vehicle_sn": {'$ne': black_sn}})
    filters = {"$and": filter_list}
    return filters, res


def get_job_objs(filters, res, service_flag=False):
    """
    Get the jobs from collection by filters
    """
    if not res.get("code"):
        res["data"] = {}
        objs = []
        if service_flag:
            objs = format_job_for_services(job_collection.find(filters))
        else:
            objs = format_job_time(job_collection.find(filters))

        sorted_objs = sorted(objs, key=lambda x: x["start_time"], reverse=True)
        res["data"]["job_objs"] = sorted_objs
        res["code"] = 200
        res["msg"] = "success"
    return res


def get_job_paginator(current_page, nums, default_pages=10):
    """
    Get the current page obj and the content index
    """
    job_paginator = paginator.Pagination(nums, default_pages)
    current_page = paginator.CurrentPaginator(current_page, job_paginator)
    first, last = current_page.get_index_content()
    return current_page, (int(first), int(last))
