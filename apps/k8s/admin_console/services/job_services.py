#!/usr/bin/env python

import datetime

import flask_restful
from flask_restful.reqparse import RequestParser

from controllers import job
from controllers import account
import fueling.common.logging as logging


class JobService(flask_restful.Resource):
    def get(self, vehicle_sn):
        parser = RequestParser()
        parser.add_argument("limit", type=int, help="one page show how many job_logs")
        parser.add_argument("offset", type=int, help="index")
        parser.add_argument("job_type", type=str, help="show job_type logs")
        parser.add_argument("starttime", type=int,
                            help="get the job log from greater than this time")
        parser.add_argument("endtime", type=int, help="get the job log from less than this time")
        args = parser.parse_args()

        limit = 50
        offset = 0
        res = {}
        filter_job = {}
        filter_endtime = []
        result = {}
        if args["limit"]:
            limit = args["limit"]
        if args["offset"]:
            offset = args["offset"]

        accounts = account.account_db.get_account_info({"vehicle_sn": vehicle_sn})
        if not accounts:
            return None, 404

        if args["endtime"]:
            filter_endtime.append({"end_time": {'$exists': False}})
            endtime = datetime.datetime.fromtimestamp(args["endtime"])
            filter_endtime.append({"end_time": {"$lte": endtime}})
            filter_job = {"$or": filter_endtime}

        filter_job["vehicle_sn"] = vehicle_sn

        if args["job_type"]:
            filter_job["job_type"] = args["job_type"]
        if args["starttime"]:
            filter_job["start_time"] = {"$gte": datetime.datetime.fromtimestamp(args["starttime"])}

        logging.info(f"filter_job: {filter_job}")
        job.get_job_objs(filter_job, res, True)
        if res["code"] != 200:
            result["desc"] = "Cannot find any job logs,please check the vehicle_sn!"
            result["total_num"] = 0
            result["result"] = []
            return result, 400
        total_num = len(res["data"]["job_objs"])
        current_page_obj, (index_start, index_end) = job.get_job_paginator(offset / limit + 1,
                                                                           total_num, limit)
        job_list = res["data"]["job_objs"][index_start: index_end]

        result["desc"] = "success"
        result["total_num"] = total_num
        result["result"] = job_list
        logging.info(f"(index_start, index_end): {(index_start, index_end)}")
        logging.info(f"res_data: {job_list}")
        return result, 200
