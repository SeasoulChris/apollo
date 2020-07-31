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
        self.partner_id = 'apollo-regression'
        self.service_url = 'http://spark-submitter-service:8000/open-service'

    def request_openservice(self, job_type, job_flags):
        """post a request to spark_submitter"""
        requestdata = {'job_type': job_type,
                       'partner': {'id': self.partner_id,
                                   'bos': {
                                        'access_key': self.ak,
                                        'secret_key': self.sk
                                   }},
                       'flags': job_flags}
        try:
            resp = requests.post(self.service_url, json=json.dumps(requestdata))
            http_code, msg = resp.status_code, resp.content
        except BaseException:
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
        logging.info(f'Regression job - {job_type} has been submitted')
        return self.request_openservice(job_type, job_flags)


if __name__ == '__main__':
    openservice = OpenserviceRegression()
    msg, http_code = openservice.submit_virtual_lane_generation()
    logging.info(f'submit_virtual_lane_generation: {http_code}: {msg}')

