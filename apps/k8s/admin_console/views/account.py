#!/usr/bin/env python3
"""
View functions for account
"""
import json
import flask

import application
from controllers import account
from fueling.common import account_utils
from fueling.common import logging
from utils import args_utils
from utils import conf_utils
from utils import time_utils

blue_account = flask.Blueprint("account", __name__,
                               template_folder="templates",
                               static_url_path="static")


@blue_account.route("/accounts", methods=["GET", "POST"])
def accounts():
    """
    account list
    """
    conf_dict = conf_utils.get_conf("ACCOUNT_SHOW_ACTION", "BLACK_LIST")
    args_dict = args_utils.get_args(("page", "1"), "vehicle_sn")
    filters = account.get_account_filter(conf_dict, args_dict)

    accounts = account.get_account_objs(filters)
    account.get_show_action(conf_dict["account_show_action"], accounts)
    current_page, index = account.get_account_paginator(args_dict["page"], len(accounts), 20)
    account_list = accounts[index[0]: index[1]]

    return flask.render_template("accounts.html", account_list=account_list,
                                 current_page=current_page, vehicle_sn=args_dict["vehicle_sn"],
                                 username=flask.session.get("user_info").get("username"))


@blue_account.route("/update_status", methods=["GET", "POST"])
def update_status():
    """
    update comment of account status.
    """
    res = {}
    data = flask.request.form

    comment = data.get("comment")
    account_id = data.get("account_id")
    action = data.get("action")
    action_type = data.get("label")
    if not comment:
        res["code"] = 201
        res["msg"] = "comment can not empty"
    elif not account_id:
        res["code"] = 202
        res["msg"] = "account_id can not empty"
    elif not action:
        res["code"] = 203
        res["msg"] = "action can not empty"
    else:
        # Todo: get the email from user account
        email = flask.session.get("user_info").get("email")
        comment_dict = account_utils.AccountUtils().\
            save_account_operation(account_id, email, comment,
                                   "account " + action_type.lower(), action)
        logging.info(f"comment_dict:{comment_dict}")
        if comment_dict:
            comment_dict["time"] = time_utils.get_datetime_str(comment_dict["time"])
            res["operation"] = comment_dict
            res["code"] = 200
            res["msg"] = "update success"
        else:
            res["code"] = 400
            res["msg"] = "update failure"
    logging.info(f"res:{res}")
    return json.dumps(res)


@blue_account.route("/edit_quota", methods=["GET", "post"])
def edit_quota():
    """
    edit the quota
    """
    package_dict = application.app.config.get("ACCOUNT_SERVICE_QUOTA")
    days_dict = application.app.config.get("ACCOUNT_SERVICE_DAYS")
    job_type = application.app.config.get("JOB_TYPE")
    res = {}
    data = flask.request.form
    selected_package = data.get("service_package")
    account_id = data["account_id"]

    account_objs = account.account_db.get_account_info({"_id": account_id})

    if not account_objs:
        res["code"] = 302
        res["msg"] = "The account id is error"
        return json.dumps(res)
    services_list = []
    for key in data:
        if key in job_type.values():
            services_list.append({"job_type": key, "status": data[key]})
    account_services = account.update_services(account_objs, services_list)
    accounts = account.get_job_used(account_services)
    if selected_package:
        if selected_package not in package_dict:
            res["code"] = 301
            res["msg"] = "The package not in services"
            return json.dumps(res)
        add_quota = package_dict[selected_package]
        accounts_add_quota = account.add_quota(accounts, add_quota)
        accounts = account.extension_date(accounts_add_quota, days_dict[selected_package])
    accounts_format_time = account.format_account_time(accounts)
    account_data = accounts_format_time[0]
    res["code"] = 200
    res["msg"] = "success"
    res["data"] = account_data
    logging.info(f"res:{res}")
    return json.dumps(res)
