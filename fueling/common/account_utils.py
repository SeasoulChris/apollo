import datetime
from bson.objectid import ObjectId

from fueling.common.mongo_utils import Mongo
import fueling.common.logging as logging


class AccountUtils(object):
    """Admin_console account utils"""

    def __init__(self):
        """Init"""
        self.db = Mongo().account_collection()

    def apply_account_info(self, com_name, com_email, vehicle_sn, bos_bucket_name, bos_region,
                           bos_ak, bos_sk, purpose, apply_date=None):
        """Apply account info"""
        date = apply_date if apply_date else datetime.datetime.now()
        result = self.db.insert_one({'com_name': com_name,
                                     'com_email': com_email,
                                     'vehicle_sn': vehicle_sn,
                                     'bos_bucket_name': bos_bucket_name,
                                     'bos_region': bos_region,
                                     'bos_ak': bos_ak,
                                     'bos_sk': bos_sk,
                                     'purpose': purpose,
                                     'status': 'Disabled',
                                     'quota': 0,
                                     'apply_date': date})
        logging.info(f"apply_account_info: {result.inserted_id}")
        return result.inserted_id

    def upadate_account_info(self, account_id, com_email, bos_bucket_name, bos_region,
                             bos_ak, bos_sk):
        """update account info"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': {'com_email': com_email,
                                     'bos_bucket_name': bos_bucket_name,
                                     'bos_region': bos_region,
                                     'bos_ak': bos_ak,
                                     'bos_sk': bos_sk}})
        logging.info(f"upadate_account_info: {account_id}")

    def update_account_status(self, account_id, status):
        """update account info"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': {'status': status}})
        logging.info(f"update_account_status: {status}")

    def save_account_quota(self, account_id, quota):
        """save account quota"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': {'quota': quota}})
        logging.info(f"save_account_quota: {account_id}")

    def save_account_services(self, account_id, services_list):
        """save account services"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': {'services': services_list}})
        logging.info(f"save_account_service: {services_list}")

    def save_account_operation(self, account_id, email, comments, action_type, status=None):
        """Save account operation"""
        update_dict = {'email': email,
                       'time': datetime.datetime.now(),
                       'comments': comments,
                       'action': {'type': action_type}}
        result = self.db.update_one({'_id': ObjectId(account_id)},
                                    {'$push': {'operations': update_dict},
                                     '$set': {'status': status}})
        logging.info(f"save_account_operation: {update_dict},"
                     f"status: {status}")
        if result.raw_result['nModified'] == 1:
            update_dict['status'] = status
            return update_dict

    def get_account_info(self, prefix):
        """get account info"""
        result = []
        if prefix.get("_id"):
            prefix["_id"] = ObjectId(prefix["_id"])
        for data in self.db.find(prefix):
            data['_id'] = data['_id'].__str__()
            result.append(data)
        logging.info(f"get_account_info num: {len(result)}")
        return result

    def save_account_due_date(self, account_id, date):
        """save account due_date"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': {'status': 'Enabled',
                                     'due_date': date}})
        logging.info(f"save_account_due_date: {account_id}")

    def create_account_msg(self, account_dict):
        """
        create account msg
        """
        if not account_dict.get("no_vehicle_sn", None):
            account_dict["no_vehicle_sn"] = False
        if not account_dict.get("apply_date", None):
            account_dict["apply_date"] = datetime.datetime.now()
        account_dict["quota"] = 0
        account_dict["status"] = "Pending"
        result = self.db.insert_one(account_dict)
        logging.info(f"apply_account_info: {result.inserted_id}")
        return result.inserted_id

    def update_account_msg(self, account_id, account_dict):
        """update account msg"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': account_dict})
        logging.info(f"update_account_msg: {account_id}")

    def save_account_operation_msg(self, account_id, operation_dict):
        """Save account operation"""
        if not operation_dict.get("time"):
            operation_dict["time"] = datetime.datetime.now()
        result = self.db.update_one({'_id': ObjectId(account_id)},
                                    {'$push': {'operations': operation_dict}})
        logging.info(f"save_account_operation: {operation_dict},")
        if result.raw_result['nModified'] == 1:
            return operation_dict


class AccountSuffixUtils(object):
    """Admin_console account suffix utils"""

    def __init__(self):
        """Init"""
        self.db = Mongo().account_suffix_collection()

    def init_suffix(self, start_num):
        self.db.insert_one({"vehicle_sn_suffix": start_num})

    def read_suffix(self):
        obj = self.db.find_one()
        if obj:
            return obj
        else:
            self.init_suffix(10)
            return self.db.find_one()

    def update_suffix(self):
        suffix_obj = self.read_suffix()
        self.db.update_one({'_id': suffix_obj["_id"]},
                           {'$set': {"vehicle_sn_suffix": suffix_obj["vehicle_sn_suffix"] + 1}})
