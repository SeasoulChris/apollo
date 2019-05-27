#!/usr/bin/env python

from http import HTTPStatus
import json
import os

from flask_restful import Resource
import flask
import google.protobuf.json_format as json_format

from modules.tools.fuel_proxy.proto.job_config_pb2 import JobConfig


class VehicleCalibration(Resource):
    """Vehicle Calibration restful service"""

    def post(self):
        """Accept user request, verify and process."""
        # 1. Parse request.
        try:
            job_config = json_format.Parse(flask.request.get_json(), JobConfig())
        except json_format.ParseError:
            return self.wrap_json(HTTPStatus.BAD_REQUEST,
                                  'Failed to parse job_config. Please verify it.')
        # 2. User authentication.
        user = 'myself'
        # 3. Data sanity check.
        # 4. Job summit.
        os.system('cd /apollo/modules/data/fuel && '
                  'nohup bash tools/submit-job-to-k8s.sh -w 1 -c 1 -m 1g -d 1 '
                  'fueling/demo/count-msg-by-channel.py &')

        return self.wrap_json(
            HTTPStatus.ACCEPTED,
            'Your job is in progress now! You will receive a notification in your corresponding '
            'email when it is finished.')

    @staticmethod
    def wrap_json(status_code, message):
        return json.dumps({'message': message}), status_code
