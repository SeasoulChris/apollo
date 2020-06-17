#!/usr/bin/env python3
"""
The logical control of the Job in front of the view
"""

import db


def get_jobs_dict():
    """
    Get the data from job return dict
    """
    jobs_dict = {"data": []}
    for job_data in db.job_db.find_all():
        jobs_dict["data"].append(job_data)
    return jobs_dict
