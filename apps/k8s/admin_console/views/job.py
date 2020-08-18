#!/usr/bin/env python3
"""
View functions for Jobs
"""

import json

import flask

from controllers import job
from fueling.common import job_utils
from utils import args_utils
from utils import conf_utils
from utils import time_utils

# Blueprint of job
blue_job = flask.Blueprint("job", __name__,
                           template_folder="templates",
                           static_url_path="static")


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


@blue_job.route("/", methods=["GET", "POST"])
@blue_job.route("/jobs", methods=["GET", "POST"])
def jobs():
    """
    job list include paginator, time filter, job-type filter, vehicle sn search ...
    """
    conf_dict = conf_utils.get_conf("JOB_TYPE", "TIME_FIELD", "SHOW_TIME_FIELD",
                                    "SHOW_JOB_TYPE", "BLACK_LIST")
    args_dict = args_utils.get_args(("page", "1"), ("job_selected", "A"),
                                    ("time_selected", "All"), "vehicle_sn")
    filters, res = job.get_job_filter(conf_dict, args_dict)
    res = job.get_job_objs(filters, res)
    current_page_obj, index_tuple = job.get_job_paginator(args_dict["page"],
                                                          len(res["data"]["job_objs"]),
                                                          20)
    if res["code"] != 200:
        return flask.render_template("error.html", error=res["msg"])
    job_list = res["data"]["job_objs"][index_tuple[0]: index_tuple[1]]
    return flask.render_template("jobs.html",
                                 jobs_list=job_list,
                                 current_page=current_page_obj,
                                 job_type=conf_dict["show_job_type"],
                                 time_field=conf_dict["show_time_field"],
                                 current_type=args_dict["job_selected"],
                                 current_time=args_dict["time_selected"],
                                 vehicle_sn=args_dict["vehicle_sn"],
                                 username=flask.session.get("user_info").get("username"))
