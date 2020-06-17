#!/usr/bin/env python3
"""
View functions for Jobs
"""

import json

import flask

import db
from controllers import job


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
    A demo function create job
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


@blue_job.route("/jobs", methods=["GET", "POST"])
def jobs():
    """
    A demo list jobs
    """
    return flask.render_template("jobs.html", jobs_dict=job.get_jobs_dict())
