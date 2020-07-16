#!/usr/bin/env python3
"""
View functions for account
"""

import flask

import application
from common import paginator
from controllers import account

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
