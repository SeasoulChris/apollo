"""open service daily regression"""

from http import HTTPStatus
import json
import os
import requests

import fueling.common.logging as logging


class OpenserviceRegression(object):
    """open service daily regression"""

    def __init__(self):
        self.ak = os.environ.get('AWS_ACCESS_KEY_ID')
        self.sk = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.vehicle_sn = 'CH0000001'
        self.bos_bucket = 'apollo-platform-fuel'
        self.bos_region = 'bj'
        self.partner_email = 'weixiao@baidu.com'
        self.service_url = 'http://spark-submitter-service:8000/open-service'

    def request_openservice(self, job_type, job_flags):
        """post a request to spark_submitter"""
        requestdata = {'job_type': job_type,
                       'partner': {'vehicle_sn': self.vehicle_sn,
                                   'email': self.partner_email,
                                   'bos': {'access_key': self.ak, 'secret_key': self.sk,
                                           'bucket': self.bos_bucket, 'region': self.bos_region}},
                       'flags': job_flags}
        logging.info(F'requestdata: {json.dumps(requestdata)}')
        try:
            resp = requests.post(self.service_url, json=json.dumps(requestdata))
            http_code, msg = resp.status_code, resp.content
            logging.info(f'Regression job - {job_type} has been submitted')
        except BaseException as ex:
            logging.error(str(ex))
            http_code = HTTPStatus.BAD_REQUEST
            msg = 'Wrong job argument'
            logging.error(F'{job_type} failed in submitting')
            logging.error(F'{msg}')
            logging.error(F'requestdata: {json.dumps(requestdata)}')
        return msg, http_code

    def submit_virtual_lane_generation(self):
        """submit virtual lane generation job"""
        job_type = 'VIRTUAL_LANE_GENERATION'
        input_data_path = 'test/openservice-regression/VirtualLaneGeneration/input'
        output_data_path = 'test/openservice-regression/VirtualLaneGeneration/output'
        job_flags = {
            'input_data_path': input_data_path,
            'zone_id': '50',
            'lidar_type': 'lidar16',
            'lane_width': '3.0',
            'extra_roi_extension': '0.5',
            'output_data_path': output_data_path
        }
        return self.request_openservice(job_type, job_flags)

    def submit_sensor_calibration(self):
        """submit sensor calibration job"""
        job_type = 'SENSOR_CALIBRATION'
        input_data_path = 'test/openservice-regression/SensorCalibration/input'
        output_data_path = 'test/openservice-regression/SensorCalibration/output'
        job_flags = {
            'input_data_path': input_data_path,
            'output_data_path': output_data_path
        }
        return self.request_openservice(job_type, job_flags)

    def submit_vehicle_calibration(self):
        """submit vehicle calibration job"""
        job_type = 'VEHICLE_CALIBRATION'
        input_data_path = 'test/openservice-regression/VehicleCalibration/input'
        job_flags = {
            'input_data_path': input_data_path,
        }
        return self.request_openservice(job_type, job_flags)

    def submit_control_profiling(self):
        """submit control profiling job"""
        job_type = 'CONTROL_PROFILING'
        input_data_path = 'test/openservice-regression/ControlProfiling/input'
        job_flags = {
            'input_data_path': input_data_path,
        }
        return self.request_openservice(job_type, job_flags)

    def submit_open_space_planner_profiling(self):
        """submit open space planner profiling job"""
        job_type = 'OPEN_SPACE_PLANNER_PROFILING'
        input_data_path = 'test/openservice-regression/OpenSpacePlannerProfiling/input'
        output_data_path = 'test/openservice-regression/OpenSpacePlannerProfiling/output'
        job_flags = {
            'input_data_path': input_data_path,
            'output_data_path': output_data_path,
        }
        return self.request_openservice(job_type, job_flags)


if __name__ == '__main__':
    openservice = OpenserviceRegression()
    jobs_to_call = ['submit_virtual_lane_generation',
                    'submit_sensor_calibration',
                    'submit_vehicle_calibration',
                    'submit_control_profiling',
                    'submit_open_space_planner_profiling']
    for job in jobs_to_call:
        msg, http_code = getattr(openservice, job)()
        logging.info(f'{job}: {http_code}: {msg}')
