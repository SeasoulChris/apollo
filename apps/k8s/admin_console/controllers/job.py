#!/usr/bin/env python3
"""
The logical control of the Job in front of the view
"""

from fueling.common import mongo_utils

job_collection = mongo_utils.Mongo().fuel_job_collection()


def get_jobs_dict(first=None, last=None):
    """
    Get the list of job to pre page
    """
    jobs = []
    for job_data in job_collection.find():
        jobs.append(job_data)
    return jobs[first:last]


def get_jobs_count():
    """
    Get the total numbers of jobs
    """
    return job_collection.count()
