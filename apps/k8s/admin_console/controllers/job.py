#!/usr/bin/env python3
"""
The logical control of the Job in front of the view
"""

from fueling.common import mongo_utils
from utils import time_utils

job_collection = mongo_utils.Mongo().fuel_job_collection()


def format_job_time(objs):
    """
    Format the job timestamp into a string
    """
    jobs = []
    for job_data in objs:
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
