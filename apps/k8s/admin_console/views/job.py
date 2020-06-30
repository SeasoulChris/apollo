#!/usr/bin/env python3
"""
View functions for Jobs
"""

import json

import flask

import application
from controllers import job
from common import paginator
from fueling.common import job_utils
from utils import time_utils


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
    res = {}
    data = flask.request.form
    # Todo: change the status of job, and update the comment of job, and add one record of history.
    if not data:
        res["code"] = 201
        res["msg"] = "data not exists"
    else:
        comment = data.get("comment")
        job_id = data.get("job_id")
        job_obj = job.get_job_by_id(job_id)
        action = data.get("action")
        # Todo: get the email from user account
        job_is_valid = job_obj.get("is_valid")
        email = "123.baidu.com"
        if (job_is_valid and action.lower() == "invalid") or \
                (not job_is_valid and action.lower() == "valid"):
            is_success = job_utils.JobUtils(job_id).\
                save_job_operations(email, comment, not job_is_valid)
            if is_success == 1:
                res["code"] = 200
                res["msg"] = "update success"
            else:
                res["code"] = 202
                res["msg"] = "update failure"
        else:
            res["code"] = 203
            res["msg"] = "Is_valid and action do not match, invalid operation"
    return json.dumps(res)


@blue_job.route("/jobs", methods=["GET", "POST"])
def jobs():
    """
    job list include paginator, time filter, job-type filter, vehicle sn search ...
    """
    # Todo: Pay attention to the capture and judgment of the value input from the front end.
    job_type = application.app.config.get("JOB_TYPE")
    time_field = application.app.config.get("TIME_FIELD")
    current_page = int(flask.request.args.get("page", 1))
    job_selected = flask.request.args.get("job_selected")
    time_selected = flask.request.args.get("time_selected")
    vehicle_sn = flask.request.args.get("vehicle_sn")
    find_filter = {}
    if job_selected and job_selected != "All":
        find_filter["job_type"] = job_selected
    if time_selected and time_selected != "All":
        days_ago = time_utils.days_ago(time_field[time_selected])
        find_filter["start_time"] = {"$gt": days_ago}
    if vehicle_sn:
        find_filter["vehicle_sn"] = vehicle_sn
    jobs_objs = job.format_job_time(job.job_collection.find(find_filter))
    job_nums = len(jobs_objs)
    job_paginator = paginator.Pagination(job_nums)
    current_page = paginator.CurrentPaginator(current_page, job_paginator)
    first, last = current_page.get_index_content()
    job_list = jobs_objs[first: last]
    return flask.render_template("jobs.html", jobs_list=job_list,
                                 current_page=current_page, job_type=job_type,
                                 time_field=time_field, current_type=job_selected,
                                 current_time=time_selected)
