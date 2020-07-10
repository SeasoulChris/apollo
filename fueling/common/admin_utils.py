from fueling.common.mongo_utils import Mongo
import fueling.common.logging as logging


class AdminUtils(object):
    """Admin_console admin utils"""

    def __init__(self):
        """Init"""
        self.db = Mongo().admin_collection()

    def save_admin_info(self, name, password, role, email):
        """Save admin info"""
        self.db.insert_one({'username': name,
                            'password': password,
                            'role': role,
                            'email': email})
        logging.info(f"save_admin_info: {email}")

    def get_admin_info(self, prefix):
        """get admin info"""
        result = []
        for data in self.db.find(prefix):
            data['_id'] = data['_id'].__str__()
            result.append(data)
        logging.info(f"get admin info: {result}")
        return result
