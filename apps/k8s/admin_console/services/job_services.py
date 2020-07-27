import json
import flask_restful
from flask_restful.reqparse import RequestParser
from controllers import job
import fueling.common.logging as logging


class JobService(flask_restful.Resource):
    def get(self, vehicle_sn):
        parser = RequestParser()
        parser.add_argument("limit", type=int, help="one page show how many job_logs")
        parser.add_argument("offset", type=int, help="index")
        parser.add_argument("job_type", type=str, help="show job_type logs")
        args = parser.parse_args()

        limit = 50
        offset = 0
        res = {}
        filter_job = {}
        result = {}
        if args["limit"]:
            limit = args["limit"]
        if args["offset"]:
            offset = args["offset"]

        filter_job["vehicle_sn"] = vehicle_sn
        if args["job_type"]:
            filter_job["job_type"] = job_type

        job.get_job_objs(filter_job, res)
        logging.info(f"vehicle_sn: {vehicle_sn}")
        total_num = len(res["data"]["job_objs"])
        current_page_obj, (index_start, index_end) = job.get_job_paginator(offset / limit + 1,
                                                                           total_num, limit)
        if res["code"] != 200:
            return json.dumps(res)
        job_list = res["data"]["job_objs"][index_start: index_end]

        result["desc"] = "success"
        result["total_num"] = total_num
        result["result"] = job_list
        logging.info(f"(index_start, index_end): {(index_start, index_end)}")
        logging.info(f"res_data: {job_list}")
        return json.dumps(result)
