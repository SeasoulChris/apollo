#!/usr/bin/env python

import flask_restful
from flask_restful import reqparse

from controllers import account
from utils import conf_utils


class AccountServices(flask_restful.Resource):
    def get(self, account_id):
        parse = reqparse.RequestParser()
        parse.add_argument("id", type=str, required=True, help="Id cannot be blank!")
        accounts = account.account_db.get_account_info({"_id": account_id})
        if not accounts:
            return {"code": 400,
                    "desc": "Don't find the account obj, please check the id",
                    "result": {}}
        accounts_used = account.get_job_used(accounts)
        accounts_objs = account.stamp_account_time(accounts_used)
        return {"code": 200, "desc": "success", "result": accounts_objs[0]}

    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument("com_name", type=str, required=True,
                           help="Com name cannot be blank!")
        parse.add_argument("com_email", type=str, required=True,
                           help="Com name cannot be blank!")
        parse.add_argument("vehicle_sn", type=str, required=False)
        parse.add_argument("bos_bucket_name", type=str, required=True,
                           help="Bos bucket name cannot be blank!")
        parse.add_argument("bos_region", type=str, required=True,
                           help="Bos region cannot be blank!")
        parse.add_argument("bos_ak", type=str, required=True,
                           help="Bos AK cannot be blank!")
        parse.add_argument("bos_sk", type=str, required=True,
                           help="Bos Sk cannot be blank!")
        parse.add_argument("purpose", type=str, required=True,
                           help="Purpose cannot be blank!")
        args = parse.parse_args()
        account_dict = args.copy()
        vehicle_sn = account_dict.get("vehicle_sn")
        account_dict["no_vehicle_sn"] = False
        if not vehicle_sn:
            account_dict["vehicle_sn"] = (account.get_virtual_vehicle_sn
                                          ("apps/k8s/admin_console/conf/admin.json"))
            account_dict["no_vehicle_sn"] = True
        account_id = account.account_db.create_account_msg(account_dict)
        conf_dict = conf_utils.get_conf("JOB_TYPE")
        job_type_list = list(conf_dict["job_type"].values())
        job_type_list.remove("All")
        services_list = []
        for job_type in job_type_list:
            services_list.append({"job_type": job_type, "status": "Disabled"})
        account.account_db.save_account_services(account_id, services_list)
        accounts = account.account_db.get_account_info({"_id": account_id})
        accounts_used = account.get_job_used(accounts)
        accounts_objs = account.stamp_account_time(accounts_used)
        return {"code": 200, "desc": "success", "result": accounts_objs[0]}

    def put(self, account_id):
        parse = reqparse.RequestParser()
        parse.add_argument("id", type=str, required=True, help="Id cannot be blank!")
        accounts = account.account_db.get_account_info({"_id": account_id})
        if not accounts:
            return {"code": 400,
                    "desc": "Don't find the account obj, please check the id",
                    "result": {}}
        account_obj = accounts[0]
        if account_obj["status"] == "Rejected":
            parse.add_argument("com_name", type=str, required=True,
                               help="Com name cannot be blank!")
            parse.add_argument("com_email", type=str, required=True,
                               help="Com name cannot be blank!")
            parse.add_argument("vehicle_sn", type=str, required=False)
            parse.add_argument("bos_bucket_name", type=str, required=True,
                               help="Bos bucket name cannot be blank!")
            parse.add_argument("bos_region", type=str, required=True,
                               help="Bos region cannot be blank!")
            parse.add_argument("bos_ak", type=str, required=True,
                               help="Bos AK cannot be blank!")
            parse.add_argument("bos_sk", type=str, required=True,
                               help="Bos Sk cannot be blank!")
            parse.add_argument("purpose", type=str, required=True,
                               help="Purpose cannot be blank!")
            args = parse.parse_args()
            account_dict = args.copy()
            account_dict.pop("id")
            vehicle_sn = account_dict.get("vehicle_sn")
            account_dict["no_vehicle_sn"] = False
            if not vehicle_sn:
                account_dict["no_vehicle_sn"] = True
                obj_vehicle_sn = account_obj.get("vehicle_sn")
                if obj_vehicle_sn:
                    is_virtual = obj_vehicle_sn.startswith("SK")
                else:
                    is_virtual = False
                account_dict["vehicle_sn"] = (obj_vehicle_sn if is_virtual else
                                              account.get_virtual_vehicle_sn
                                              ("apps/k8s/admin_console/conf/admin.json"))
            account.account_db.update_account_msg(account_id, account_dict)
        elif account_obj["status"] == "Enabled":
            parse.add_argument("com_name", type=str, required=True,
                               help="Com name cannot be blank!")
            parse.add_argument("com_email", type=str, required=True,
                               help="Com name cannot be blank!")
            parse.add_argument("bos_bucket_name", type=str, required=True,
                               help="Bos bucket name cannot be blank!")
            parse.add_argument("bos_ak", type=str, required=True,
                               help="Bos AK cannot be blank!")
            parse.add_argument("bos_sk", type=str, required=True,
                               help="Bos Sk cannot be blank!")
            args = parse.parse_args()
            account_dict = args.copy()
            account_dict.pop("id")
            account.account_db.update_account_msg(account_id, account_dict)
        else:
            return {"code": 400,
                    "desc": "The status is error, the data hasn't changed",
                    "result": {}}
        new_accounts = account.account_db.get_account_info({"_id": account_id})
        accounts_used = account.get_job_used(new_accounts)
        accounts_objs = account.stamp_account_time(accounts_used)
        return {"code": 200, "desc": "success", "result": accounts_objs[0]}
