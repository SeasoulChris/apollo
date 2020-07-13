import datetime
from bson.objectid import ObjectId

from fueling.common.mongo_utils import Mongo
import fueling.common.logging as logging


class AccountUtils(object):
    """Admin_console account utils"""

    def __init__(self):
        """Init"""
        self.db = Mongo().account_collection()

    def apply_account_info(self, com_name, com_email, vehicle_sn, bos_bucker_name, bos_region,
                           bos_ak, bos_sk, purpose, apply_date=None):
        """Apply account info"""
        date = apply_date if apply_date else datetime.datetime.now()
        result = self.db.insert_one({'com_name': com_name,
                                     'com_email': com_email,
                                     'vehicle_sn': vehicle_sn,
                                     'bos_bucker_name': bos_bucker_name,
                                     'bos_region': bos_region,
                                     'bos_ak': bos_ak,
                                     'bos_sk': bos_sk,
                                     'purpose': purpose,
                                     'status': 'Disabled',
                                     'quota': 0,
                                     'apply_date': date})
        logging.info(f"apply_account_info: {result.inserted_id}")
        return result.inserted_id

    def upadate_account_info(self, account_id, com_email, bos_bucker_name, bos_region,
                             bos_ak, bos_sk):
        """update account info"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': {'com_email': com_email,
                                     'bos_bucker_name': bos_bucker_name,
                                     'bos_region': bos_region,
                                     'bos_ak': bos_ak,
                                     'bos_sk': bos_sk}})
        logging.info(f"upadate_account_info: {account_id}")

    def save_account_quota(self, account_id, quota):
        """save account quota"""
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$set': {'quota': quota}})
        logging.info(f"save_account_quota: {account_id}")

    def save_account_services(self, account_id, job_type_list, status, due_date=None):
        """save account services"""
        services = []
        for job_type in job_type_list:
            services.append({'job_type': job_type, 'status': status})
        if status == 'Enabled':
            date = due_date if due_date else datetime.datetime.now() + datetime.timedelta(days=365)
            self.db.update_one({'_id': ObjectId(account_id)},
                               {'$set': {'status': 'Enabled',
                                         'due_date': date,
                                         'services': services}})
        else:
            self.db.update_one({'_id': ObjectId(account_id)},
                               {'$set': {'status': status,
                                         'services': services}})
        logging.info(f"save_account_service: {services}")

    def save_account_operation(self, account_id, email, comments, action_type):
        """Save account operation"""
        update_dict = {'email': email,
                       'time': datetime.datetime.now(),
                       'comments': comments,
                       'action': {'type': action_type}}
        self.db.update_one({'_id': ObjectId(account_id)},
                           {'$push': {'operations': update_dict}})
        logging.info(f"save_account_operation: {operations}")

    def get_account_info(self, account_id):
        """get account info"""
        result = []
        for data in self.db.find({'_id': ObjectId(account_id)}):
            data['_id'] = data['_id'].__str__()
            result.append(data)
        logging.info(f"get_account_info: {result}")
        return result
