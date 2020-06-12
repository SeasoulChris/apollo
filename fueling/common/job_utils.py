#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""fuel job utils"""

from datetime import datetime

from fueling.common.mongo_utils import Mongo
import fueling.common.record_utils as record_utils


class JobUtils(object):
    """fuel job utils"""

    def __init__(self, job_id):
        """Init"""
        self.job_id = job_id
        self.db = Mongo().fuel_job_collection()

    def save_job_submit_info(self, is_partner):
        """Save job submit info"""
        self.db.insert_one({'job_id': self.job_id,
                            'is_partner': is_partner,
                            'start_time': datetime.now(),
                            'status': 'Running',
                            'progress': 0})

    def save_job_location(self, filename):
        """Save job location to mongodb"""
        msg_reader = record_utils.read_record(
            [record_utils.LOCALIZATION_CHANNEL])
        for msg in msg_reader(filename):
            localization = record_utils.message_to_proto(msg)
            if localization is None:
                return
            pose = localization.pose
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            # Todo (weixiao)
            break

    def save_job_progress(self, progress):
        """Save job running progress
        progress should be in the range [0, 100]
        """
        if not progress >= 0 or not progress <= 100:
            return
        self.db.update({'job_id': self.job_id},
                       {'$set': {'progress': progress}})

    def save_job_operations(self, email, comments, is_valid):
        """Save job operations
        """
        action_type = 'valid' if is_valid else 'invalid'
        self.db.update({'job_id': self.job_id},
                       {'$push': {'operations':
                                  {'email': email,
                                   'time': datetime.now(),
                                   'comments': comments,
                                   'action': {'type': action_type}}},
                        '$set': {'is_valid': is_valid}})

    def save_job_phase(self, status):
        """Save job status
        The status value will in ['Succeeded', 'Failed', 'Running']
        """
        if status == 'Succeeded':
            self.db.update_one({'job_id': self.job_id},
                               {'$set': {'status': status,
                                         'end_time': datetime.now(),
                                         'progress': 100}})
        else:
            self.db.update_one({'job_id': self.job_id},
                               {'$set': {'status': status,
                                         'end_time': datetime.now()}})
