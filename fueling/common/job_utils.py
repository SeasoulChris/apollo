#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""fuel job utils"""

from datetime import datetime
import math

from pyproj import Proj

from fueling.common.mongo_utils import Mongo
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


def transform_utm_to_lat_lon(x, y, zone_id=50, hemisphere='N'):
    """ transform utm coordinates to longitude and latitude
    hemisphere will be in ('N', 'S')
    default zone_id=50: Beijing
    """
    h_north = False
    h_south = False
    if (hemisphere == 'N'):
        h_north = True
    elif (hemisphere == 'S'):
        h_south = True

    proj_in = Proj(proj='utm', zone=zone_id, ellps='WGS84', south=h_south, north=h_north,
                   errcheck=True)

    longitude, latitude = proj_in(x, y, inverse=True)

    longitude = math.floor(longitude * 1000000) / 1000000
    latitude = math.floor(latitude * 1000000) / 1000000

    return longitude, latitude


class JobUtils(object):
    """fuel job utils"""

    def __init__(self, job_id):
        """Init"""
        self.job_id = job_id
        self.db = Mongo().fuel_job_collection()
        logging.info("Init Job Utils")

    def save_job_submit_info(self):
        """Save job submit info"""
        self.db.insert_one({'job_id': self.job_id,
                            'is_valid': True,
                            'start_time': datetime.now(),
                            'status': 'Running',
                            'progress': 0})

    def save_job_location(self, filename, zone_id=50, hemisphere='N'):
        """Save job location to mongodb"""
        msg_reader = record_utils.read_record(
            [record_utils.LOCALIZATION_CHANNEL])
        for msg in msg_reader(filename):
            localization = record_utils.message_to_proto(msg)
            if localization is None:
                return
            pose = localization.pose
            x, y = pose.position.x, pose.position.y
            longitude, latitude = transform_utm_to_lat_lon(x, y, zone_id, hemisphere)
            self.db.update_one({'job_id': self.job_id},
                               {'$set': {'localization': {'x': x,
                                                          'y': y,
                                                          'zone_id': zone_id}}})
            break

    def save_job_progress(self, progress):
        """Save job running progress
        progress should be in the range [0, 100]
        """
        if not progress >= 0 or not progress <= 100:
            return
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'progress': progress}})

    def save_job_operations(self, email, comments, is_valid):
        """Save job operations
        """
        action_type = 'valid' if is_valid else 'invalid'
        self.db.update_one({'job_id': self.job_id},
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
