#!/usr/bin/env python

from http import HTTPStatus
import json
import os


class VehicleCalibration(object):
    """Vehicle Calibration restful service"""

    def __init__(self, job_config):
        self.job_config = job_config

    def process(self):
        """Accept user request, verify and process."""
        # TODO: Data sanity check.
        # TODO: Job summit.
        os.system('cd /apollo/modules/data/fuel && '
                  'nohup bash tools/submit-job-to-k8s.sh -w 1 -c 1 -m 1g -d 1 '
                  'fueling/demo/count-msg-by-channel.py &')

        return json.dumps({
            'message': 'Your job is in progress now! You will receive a notification in your corresponding '
            'email when it is finished.'}
        ), HTTPStatus.ACCEPTED
