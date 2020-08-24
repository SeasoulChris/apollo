"""open service daily regression"""

from collections import defaultdict as df
from collections import namedtuple as nt
from datetime import datetime
from http import HTTPStatus
import json
import os
import requests
import time

import fueling.common.email_utils as email_utils
import fueling.common.logging as logging


class OpenserviceRegression(object):
    """open service daily regression"""

    def __init__(self):
        """init"""
        self.ak = os.environ.get('AWS_ACCESS_KEY_ID')
        self.sk = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.vehicle_sn = 'CH0000001'
        self.bos_bucket = 'apollo-platform-fuel'
        self.bos_region = 'bj'
        self.partner_email = 'weixiao@baidu.com'
        self.service_host = 'http://spark-submitter-service:8000'
        self.service_url = F'{self.service_host}/open-service'

    def get_job_status(self, job_id):
        """get job status"""
        res = requests.get(self.service_host, params={'job_id': job_id})
        if res.ok:
            return json.loads(res.json() or '{}').get('status')

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
            msg = msg.decode('utf8').strip()
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
    error_label = False
    job_info = df(lambda: df(lambda: ''))
    submited_cnt = 0
    # kick off
    for job in jobs_to_call:
        msg, http_code = getattr(openservice, job)()
        if ':' in msg:
            job_id = msg.split(':')[0].strip('"')
            job_info[job]['job_id'] = job_id
            submited_cnt += 1
            job_info[job]['start_time'] = datetime.now()
            logging.info(f'{job_id}: {http_code}: {msg}')
        else:
            logging.error(f'{http_code}: {msg}')
            error_label = True
        time.sleep(5)

    # monitor
    WAIT_INTERVAL_SECONDS = 3
    END_STATUS = {'Completed', 'Error', 'UnexpectedAdmissionError', 'Terminating'}
    completed_cnt = 0
    start_time = datetime.now()
    max_checking_seconds = 3600 * 10
    while True:
        for job in jobs_to_call:
            job_id = job_info[job]['job_id']
            if job_id and not job_info[job]['end_time']:
                status = openservice.get_job_status(job_id)
                logging.info(F'{job}:{job_id}:{status}')
                job_info[job]['status'] = status
                if status in END_STATUS:
                    job_info[job]['end_time'] = datetime.now()
                    seconds = (job_info[job]['end_time'] - job_info[job]['start_time'])\
                        .total_seconds()
                    hours = seconds // 3600
                    minutes = (seconds // 60) - (hours * 60)
                    duration = F'{hours} hour {minutes} min'
                    job_info[job]['duration'] = duration
                    logging.info(F'{job}:{job_id}: runningtime: {duration}')
                    completed_cnt += 1
        # all jobs are done
        if submited_cnt == completed_cnt:
            logging.info('Checking completed successfully')
            break
        # running time is out of limit
        current_time = datetime.now()
        checking_seconds = (current_time - start_time).total_seconds()
        if checking_seconds > max_checking_seconds:
            error_label = True
            logging.error('Checking time exceeds max limit')
            break
        time.sleep(WAIT_INTERVAL_SECONDS)

    # send report
    reports = []
    job_tuple = nt('job_tuple', 'job status job_id start end duration')
    for job in jobs_to_call:
        job_id = job_info[job].get('job_id', '-')
        start_time = job_info[job].get('start_time', '-').strftime("%m/%d/%Y %H:%M:%S")
        end_time = job_info[job].get('end_time', '-').strftime("%m/%d/%Y %H:%M:%S")
        duration = job_info[job].get('duration', '-')
        status = job_info[job].get('status', '-')
        job_name = job.lstrip('submit_')
        if status != 'Completed':
            error_label = True
        reports.append(job_tuple(job_name, status, job_id, start_time, end_time, duration))
    title = 'Open Service Daily Regression Report'
    content = tuple(reports)
    logging.info(content)
    receivers = email_utils.DATA_TEAM + email_utils.D_KIT_TEAM
    if error_label:
        email_utils.send_email_error(title, content, receivers)
    else:
        email_utils.send_email_info(title, content, receivers)
