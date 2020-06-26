#!/usr/bin/env python3
"""
View functions for Jobs
"""

import json

import flask

from controllers import job
from common import paginator


# Blueprint of job
blue_job = flask.Blueprint("job", __name__,
                           template_folder="templates",
                           static_url_path="static")


@blue_job.route("/jobs_json", methods=["GET", "POST"])
def jobs_json():
    """
    A demo function return jobs of json
    """
    jobs_dict = job.get_jobs_dict()
    return flask.Response(json.dumps(jobs_dict), content_type="application/json")


@blue_job.route('/create_job', methods=["POST", "GET"])
def create_job():
    """
    A demo function create job.
    Note：This is test code and should not be called in production
    """
    db.job_db.insert(
        {
            "job_type": "Vehicle Calibration",
            "job_id": "0001",
            "vehicle_sn": "qwqw",
            "status": "success",
            "start_time": "2020-06-01 8:00",
            "end_time": "2020-06-01 11:00",
            "is_valid": "true",
            "input_data_size": "/home",
            "localization": "beijing",
            "progress": "100%",
            "operation": [
                {
                    "operator_email": "abc@baidu.com",
                    "operator_time": "2020-06-01 14:00",
                    "comments": "this is a message",
                    "action": "isvalid",
                }
            ]
        }
    )
    return flask.redirect("/jobs")


@blue_job.route("/submit_job", methods=["GET", "POST"])
def submit_job():
    """
    Submit comment of job where admin is_valid or valid.
    """
    print(flask.request.form.get("data"), type(flask.request.form))
    # Todo: change the status of job, and update the comment of job, and add one record of history.
    return json.dumps({"code": 200})


@blue_job.route("/jobs", methods=["GET", "POST"])
def jobs():
    """
    A demo list jobs
    """
    # Todo: Pay attention to the capture and judgment of the value input from the front end.
    current_page = int(flask.request.args.get("page", 1))
    job_nums = job.get_jobs_count()
    job_paginator = paginator.Pagination(job_nums)
    current_page = paginator.CurrentPaginator(current_page, job_paginator)
    first, last = current_page.get_index_content()
    job_list = job.get_jobs_dict(first, last)
    return flask.render_template("jobs.html", jobs_list=job_list, current_page=current_page)
