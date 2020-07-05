#!/usr/bin/env python
# -*- coding: UTF-8-*-
"""fuel job utils"""

from datetime import datetime
import math

from pyproj import Proj

from fueling.common.mongo_utils import Mongo
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


def transform_utm_to_lat_lon(x, y, zone_id=50, hemisphere='N'):
    """ transform utm coordinates to longitude and latitude
    hemisphere will be in ('N', 'S')
    default zone_id=50: Beijing
    """
    h_north = False
    h_south = False
    if hemisphere == 'N':
        h_north = True
    elif hemisphere == 'S':
        h_south = True

    proj_in = Proj(proj='utm', zone=zone_id, ellps='WGS84', south=h_south, north=h_north,
                   errcheck=True)

    longitude, latitude = proj_in(x, y, inverse=True)

    longitude = math.floor(longitude * 1000000) / 1000000
    latitude = math.floor(latitude * 1000000) / 1000000

    return longitude, latitude


def get_jobs_list():
    """get job list info"""
    result = []
    for job_data in Mongo().fuel_job_collection().find():
        job_data['_id'] = job_data['_id'].__str__()
        result.append(job_data)
    logging.info(f"get job list num: {len(result)}")
    return result


def extract_flags(flags):
    """extract flags string to dict"""
    flags_list = flags.split('--')
    flags_dict = {}
    for item in flags_list:
        if '=' in item:
            key, val = item.strip().split('=')
            flags_dict[key.strip()] = val.strip()
    return flags_dict


class JobUtils(object):
    """fuel job utils"""

    def __init__(self, job_id):
        """Init"""
        self.job_id = job_id
        self.db = Mongo().fuel_job_collection()

    def save_job_submit_info(self):
        """Save job submit info"""
        self.db.insert_one({'job_id': self.job_id,
                            'is_valid': True,
                            'start_time': datetime.now(),
                            'status': 'Running',
                            'progress': 0})
        logging.info(f"save_job_submit_info: {self.job_id}")

    def save_job_vehicle_sn(self, vehicle_sn):
        """Save job vehicle_sn"""
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'vehicle_sn': vehicle_sn}})
        logging.info(f"save_job_vehicle_sn: {vehicle_sn}")

    def save_job_partner(self, is_partner):
        """Save job partner label (boolean)"""
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'is_partner': is_partner}})
        logging.info(f"save_job_partner: {is_partner}")

    def save_job_type(self, job_type):
        """Save job type info"""
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'job_type': job_type}})
        logging.info(f"save_job_type: {job_type}")

    def save_job_sub_type(self, sub_type):
        """Save job sub_type"""
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'sub_type': sub_type}})
        logging.info(f"save_job_sub_type: {sub_type}")

    def save_job_input_data_size(self, source_dir):
        """Save job input data size"""
        input_data_size = file_utils.getDirSize(source_dir)
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'input_data_size': input_data_size}})
        logging.info(f"save_job_input_data_size: {source_dir}: {input_data_size}")

    def save_job_location(self, x, y, zone_id=50, hemisphere='N'):
        """Save job location to mongodb
        params: UTM-Coordinates
        """
        longitude, latitude = transform_utm_to_lat_lon(x, y, zone_id, hemisphere)
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'localization': {'x': x,
                                                      'y': y,
                                                      'longitude': longitude,
                                                      'latitude': latitude,
                                                      'zone_id': zone_id}}})
        logging.info(f"save_job_location: x: {x}, y: {y}, "
                     f"zone_id: {zone_id}"
                     f"longitude: {longitude}"
                     f"latitude: {latitude}")
        return longitude, latitude

    def save_job_progress(self, progress):
        """Save job running progress
        progress should be in the range [0, 100]
        """
        if not progress >= 0 or not progress <= 100:
            return
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'progress': progress}})
        logging.info(f"save_job_progress: {progress}")

    def save_job_operations(self, email, comments, is_valid):
        """Save job operations
        """
        action_type = 'valid' if is_valid else 'invalid'
        update_dict = {'email': email,
                       'time': datetime.now(),
                       'comments': comments,
                       'action': {'type': action_type}}
        result = self.db.update_one({'job_id': self.job_id},
                                    {'$push': {'operations': update_dict},
                                     '$set': {'is_valid': is_valid}})
        logging.info(f"save_job_operations: email: {email},"
                     f"comments: {comments},"
                     f"is_valid: {is_valid}")
        if result.raw_result['nModified'] == 1:
            update_dict['is_valid'] = is_valid
            return update_dict

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
        logging.info(f"save_job_phase: {status}")

    def save_job_failure_code(self, err_code):
        """Save job err_code"""
        self.db.update_one({'job_id': self.job_id},
                           {'$set': {'failure_code': err_code}})
        logging.info(f"save_job_failure_code: {err_code}")

    def get_job_info(self):
        """get job info"""
        result = []
        for job_data in self.db.find({'job_id': self.job_id}):
            job_data['_id'] = job_data['_id'].__str__()
            result.append(job_data)
        logging.info(f"get job info result: {result}")
        return result
