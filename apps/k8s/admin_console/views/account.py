#!/usr/bin/env python3
"""
View functions for account
"""
import json
import flask

import application
from common import paginator
from controllers import account
from fueling.common import account_utils
from fueling.common import logging
from utils import time_utils

blue_account = flask.Blueprint("account", __name__,
                               template_folder="templates",
                               static_url_path="static")


@blue_account.route("/accounts", methods=["GET", "POST"])
def accounts():
    """
    account list
    """
    black_list = application.app.config.get("BLACK_LIST")
    account_show_action = application.app.config.get("ACCOUNT_SHOW_ACTION")
    current_page = int(flask.request.args.get("page", 1))
    vehicle_sn = flask.request.args.get("vehicle_sn")
    find_filter = []
    if vehicle_sn:
        find_filter.append({"vehicle_sn": vehicle_sn})
    else:
        for black_sn in black_list:
            find_filter.append({"vehicle_sn": {'$ne': black_sn}})
    if find_filter:
        filters = {"$and": find_filter}
    else:
        filters = {}
    accounts = account.account_db.get_account_info(filters)
    account_used = account.get_job_used(accounts)
    account_objs = account.format_account_time(account_used)
    account_add_show_actions = account.get_show_action(account_show_action, account_objs)
    account_nums = len(account_add_show_actions)
    account_paginator = paginator.Pagination(account_nums, 20)
    current_page = paginator.CurrentPaginator(current_page, account_paginator)
    first, last = current_page.get_index_content()
    account_list = sorted(account_objs, key=lambda x: x["due_date"])[first: last]
    return flask.render_template("accounts.html", account_list=account_list,
                                 current_page=current_page, vehicle_sn=vehicle_sn,
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
    res = {}
    data = flask.request.form
    selected_package = data["service_package"]
    account_id = data["account_id"]
    if selected_package not in package_dict:
        res["code"] = 201
        res["msg"] = "The package not in services"
        return json.dumps(res)
    add_quota = package_dict[selected_package]
    account_objs = account.account_db.get_account_info({"_id": account_id})
    if not account_objs:
        res["code"] = 202
        res["msg"] = "The account id is error"
        return json.dumps(res)
    accounts_add_used = account.get_job_used(account_objs)
    accounts_add_quota = account.add_quota(accounts_add_used, add_quota)
    accounts_add_due_date = account.extension_date(accounts_add_quota, days_dict[selected_package])
    accounts_format_time = account.format_account_time(accounts_add_due_date)
    account_data = accounts_format_time[0]
    res["code"] = 200
    res["msg"] = "success"
    res["data"] = account_data
    return json.dumps(res)
