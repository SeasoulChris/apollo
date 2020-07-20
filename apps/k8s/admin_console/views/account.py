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

    """
    black_list = application.app.config.get("BLACK_LIST")
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
    account_nums = len(account_objs)
    account_paginator = paginator.Pagination(account_nums, 20)
    current_page = paginator.CurrentPaginator(current_page, account_paginator)
    first, last = current_page.get_index_content()
    account_list = sorted(account_objs, key=lambda x: x["apply_date"], reverse=True)[first: last]
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
