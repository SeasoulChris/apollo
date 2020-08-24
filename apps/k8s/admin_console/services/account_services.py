import flask
import flask_restful

from controllers import account
from utils import conf_utils


class AccountServices(flask_restful.Resource):
    def get(self, account_id):
        if len(account_id) != 24:
            return None, 404
        accounts = account.account_db.get_account_info({"_id": account_id})
        if not accounts:
            return None, 404
        accounts_used = account.get_job_used(accounts)
        accounts_objs = account.stamp_account_time(accounts_used)
        return {"desc": "success", "result": accounts_objs[0]}

    def post(self):
        must_args = ["com_name", "com_email",
                     "bos_bucket_name", "bos_region",
                     "bos_ak", "bos_sk", "purpose"]
        args = flask.request.form
        account_dict = {}
        message_dict = {"message": {}}
        for key in args:
            account_dict[key] = args.get(key)
        for arg in must_args:
            if arg not in account_dict:
                message_dict["message"]["details"] = "{} cannot be empty!".format(arg)
                message_dict["message"]["code"] = "E01"
                message_dict["message"]["fields"] = arg
                return message_dict, 400
        vehicle_sn = account_dict.get("vehicle_sn")
        account_dict["no_vehicle_sn"] = False
        if not vehicle_sn:
            account_dict["vehicle_sn"] = (account.get_virtual_vehicle_sn())
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
        return {"desc": "success", "result": accounts_objs[0]}

    def put(self, account_id):
        if len(account_id) != 24:
            return None, 404
        accounts = account.account_db.get_account_info({"_id": account_id})
        if not accounts:
            return None, 404
        must_args = ["com_name", "com_email",
                     "bos_bucket_name",
                     "bos_ak", "bos_sk"]
        args = flask.request.form
        account_dict = {}
        for key in args:
            account_dict[key] = args.get(key)
        account_obj = accounts[0]
        message_dict = {"message": {}}
        if account_obj["status"] == "Rejected":
            must_args.extend(["bos_region", "purpose"])
            for arg in must_args:
                if arg not in account_dict:
                    message_dict["message"]["details"] = "{} cannot be empty!".format(arg)
                    message_dict["message"]["code"] = "E01"
                    message_dict["message"]["fields"] = arg
                    return message_dict, 400
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
                                              account.get_virtual_vehicle_sn())
            account.account_db.update_account_msg(account_id, account_dict)
            account.account_db.upadate_account_status(account_id, 'Pending')
        elif account_obj["status"] == "Enabled":
            for arg in must_args:
                if arg not in account_dict:
                    message_dict["message"]["details"] = "{} cannot be empty!".format(arg)
                    message_dict["message"]["code"] = "E01"
                    message_dict["message"]["fields"] = arg
                    return message_dict, 400
            account.account_db.update_account_msg(account_id, account_dict)
        else:
            message_dict["message"]["details"] = (
                "Account data can't be updated in current account status"
            )
            message_dict["message"]["code"] = "E02"
            return message_dict, 400
        new_accounts = account.account_db.get_account_info({"_id": account_id})
        accounts_used = account.get_job_used(new_accounts)
        accounts_objs = account.stamp_account_time(accounts_used)
        return {"desc": "success", "result": accounts_objs[0]}
