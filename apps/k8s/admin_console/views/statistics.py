#!/usr/bin/env python3
"""
View functions for Statistics
"""

import flask

import application
from controllers import job
from fueling.common import job_utils
from utils import time_utils
import fueling.common.logging as logging


# Blueprint of statistics
blue_statistics = flask.Blueprint("statistics", __name__,
                                  template_folder="templates",
                                  static_url_path="static")


@blue_statistics.route('/statistics', methods=["GET", "POST"])
def statistics():
    """
    The fuction of statistics
    """
    job_type_filed = application.app.config.get("JOB_TYPE")
    time_field = application.app.config.get("TIME_FIELD")
    show_time_field = application.app.config.get("SHOW_TIME_FIELD")
    show_job_type = application.app.config.get("SHOW_JOB_TYPE")
    black_list = application.app.config.get("BLACK_LIST")
    show_aggregated_by = application.app.config.get("AGGREGATED_FIELD")
    aggregated_by = application.app.config.get("AGGREGATED_BY")
    job_selected = flask.request.args.get("job_selected", "A")
    time_selected = flask.request.args.get("time_selected")
    vehicle_sn = flask.request.args.get("vehicle_sn")
    aggregated_selected = flask.request.args.get("aggregated_selected")

    find_filter = {}
    labels = []
    weeks = []
    num_list = {}
    num_list_temp = {}
    show_job_type_list = {}
    job_type_list = []
    labels_aggregated = []
    aggregated_filed = {}
    selc_aggregated = ""

    find_filter["is_partner"] = True
    # find_filter["is_valid"] = True

    if vehicle_sn:
        find_filter["vehicle_sn"] = vehicle_sn
    else:
        for black_sn in black_list:
            find_filter["vehicle_sn"] = {'$ne': black_sn}

    if aggregated_selected:
        selc_aggregated = aggregated_by[aggregated_selected]
    else:
        selc_aggregated = "week"

    if job_selected:
        if job_selected not in job_selected:
            return flask.render_template("error.html", error="Invalid parameter")
        elif job_selected != "A":
            job_type_list = [job_selected]
            show_job_type_list = {job_selected: show_job_type[job_selected]}
        else:
            job_type_list = list(show_job_type.keys())
            show_job_type_list = show_job_type.copy()
    else:
        job_type_list = list(show_job_type.keys())
        show_job_type_list = show_job_type.copy()

    aggregated_filed = show_aggregated_by

    logging.info(f"job_selected: {job_selected}")
    if time_selected:
        if time_selected not in time_field:
            return flask.render_template("error.html", error="Invalid parameter")
        elif time_selected != "All":
            if time_field[time_selected] == 7:
                aggregated_filed = {"W": "周"}
            elif time_field[time_selected] == 30:
                aggregated_filed = {"W": "周", "M": "月"}
            else:
                aggregated_filed = show_aggregated_by
            days_ago = time_utils.days_ago(time_field[time_selected])
            find_filter["start_time"] = {"$gt": days_ago}
            selc_aggregated = aggregated_by[aggregated_selected]
    else:
        aggregated_filed = show_aggregated_by
        selc_aggregated = "week"

    labels.append('任务类型')
    logging.info(f"time_selected: {time_selected}")
    logging.info(f"selc_aggregated: {selc_aggregated}")
    logging.info(f"job_type_list: {job_type_list}")
    job_type_list.sort(reverse=True)
    list_dict = sorted(show_job_type_list.items(),
                       key=lambda show_job_type_list: show_job_type_list[0],
                       reverse=True)
    show_job_type_list.clear()
    for temp in list_dict:
        show_job_type_list[temp[0]] = temp[1]
    for job_type in job_type_list:
        num_list_temp[job_type] = {}
        if job_type != "A":
            find_filter["job_type"] = job_type_filed[job_type]
        else:
            find_filter["job_type"] = {"$in": [job_type_filed[type_job]
                                               for type_job in job_type_list if type_job != "A"]}
        if not selc_aggregated:
            selc_aggregated = "week"
        aggregated = f"$" + selc_aggregated
        logging.info(f"find_filter: {find_filter}")

        operator = []
        if selc_aggregated != "year":
            operator = [{"$match": find_filter},
                        {"$group": {"_id": {selc_aggregated: {aggregated: "$start_time"},
                                            "year": {"$year": "$start_time"}},
                                    "count": {"$sum": 1}}},
                        {"$sort": {"_id": 1}}]
        else:
            operator = [{"$match": find_filter},
                        {"$group": {"_id": {aggregated: "$start_time"},
                                    "count": {"$sum": 1}}},
                        {"$sort": {"_id": 1}}]

        subquery = job.job_collection.aggregate(operator)
        results = [doc for doc in subquery]
        logging.info(f"get aggregated: {results}")
        for doc in results:
            labedate = ''
            if selc_aggregated == "week":
                weekdate = str(doc['_id']["year"]) + str(doc['_id']["week"])
                labedate = time_utils.get_first_day(weekdate)
            elif selc_aggregated == "month":
                if doc['_id']["month"] < 10:
                    labedate = str(doc['_id']["year"]) + "-0" + str(doc['_id']["month"])
                else:
                    labedate = str(doc['_id']["year"]) + "-" + str(doc['_id']["month"])
            else:
                labedate = doc['_id']

            if labedate not in labels_aggregated:
                labels_aggregated.append(labedate)
            num_list_temp[job_type][labedate] = doc['count']

        logging.info(f"get labels_aggregated: {labels_aggregated}")
        logging.info(f"get num_list_temp: {num_list_temp}")
    labels_aggregated.sort()
    labels.extend(labels_aggregated)

    for job_type in job_type_list:
        num_list[job_type] = []
        for label in labels_aggregated:
            if label not in num_list_temp[job_type]:
                num_list_temp[job_type][label] = 0
            num_list[job_type].append(num_list_temp[job_type][label])
    logging.info(f"get new labels_aggregated: {labels_aggregated}")
    logging.info(f"get new num_list: {num_list_temp}")
    logging.info(f"get num_list: {num_list}")
    return flask.render_template('statistics.html', lable_list=labels,
                                 labels_aggregated=labels_aggregated, num_list=num_list,
                                 job_type=show_job_type, job_type_list=show_job_type_list,
                                 time_field=show_time_field, aggregated_by=aggregated_filed,
                                 current_type=job_selected, current_time=time_selected,
                                 current_aggregated=aggregated_selected)
