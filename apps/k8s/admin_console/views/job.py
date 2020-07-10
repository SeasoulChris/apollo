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
    comment = data.get("comment")
    job_id = data.get("job_id")
    action = data.get("action")
    if not comment:
        res["code"] = 201
        res["msg"] = "comment can not empty"
    elif not job_id:
        res["code"] = 202
        res["msg"] = "job_id can not empty"
    elif not action:
        res["code"] = 203
        res["msg"] = "action can not empty"
    else:
        job_obj = job.get_job_by_id(job_id)
        # Todo: get the email from user account
        job_is_valid = job_obj.get("is_valid")
        email = flask.session.get("user_info").get("email")
        if ((job_is_valid and action.lower() == "invalid")
                or (not job_is_valid and action.lower() == "valid")):
            comment_dict = job_utils.JobUtils(job_id).\
                save_job_operations(email, comment, not job_is_valid)
            if comment_dict:
                comment_dict["time"] = time_utils.get_datetime_str(comment_dict["time"])
                res["operation"] = comment_dict
                res["code"] = 200
                res["msg"] = "update success"
            else:
                res["code"] = 400
                res["msg"] = "update failure"
        else:
            res["code"] = 300
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
    show_time_field = application.app.config.get("SHOW_TIME_FIELD")
    show_job_type = application.app.config.get("SHOW_JOB_TYPE")
    black_list = application.app.config.get("BLACK_LIST")
    current_page = int(flask.request.args.get("page", 1))
    job_selected = flask.request.args.get("job_selected", "A")
    time_selected = flask.request.args.get("time_selected", "All")
    vehicle_sn = flask.request.args.get("vehicle_sn")
    find_filter = [{"is_partner": True}]
    if job_selected not in job_type:
        return flask.render_template("error.html", error="Invalid parameter")
    elif job_selected != "A":
        find_filter.append({"job_type": job_type[job_selected]})
    if time_selected not in time_field:
        return flask.render_template("error.html", error="Invalid parameter")
    elif time_selected != "All":
        days_ago = time_utils.days_ago(time_field[time_selected])
        find_filter.append({"start_time": {"$gt": days_ago}})
    if vehicle_sn:
        find_filter.append({"vehicle_sn": vehicle_sn})
    else:
        for black_sn in black_list:
            find_filter.append({"vehicle_sn": {'$ne': black_sn}})
    if find_filter:
        filters = {"$and": find_filter}
    else:
        filters = {}
    jobs_objs = job.format_job_time(job.job_collection.find(filters))
    job_nums = len(jobs_objs)
    job_paginator = paginator.Pagination(job_nums, 20)
    current_page = paginator.CurrentPaginator(current_page, job_paginator)
    first, last = current_page.get_index_content()
    job_list = sorted(jobs_objs, key=lambda x: x["start_time"], reverse=True)[first: last]
    return flask.render_template("jobs.html", jobs_list=job_list,
                                 current_page=current_page, job_type=show_job_type,
                                 time_field=show_time_field, current_type=job_selected,
                                 current_time=time_selected, vehicle_sn=vehicle_sn,
                                 username=flask.session.get("user_info").get("username"))
