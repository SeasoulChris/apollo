#!/usr/bin/env python

import json
from http import HTTPStatus

import flask
from flask import request
from absl import flags
from absl import logging
from absl import app as absl_app
import requests

from modules.data.fuel.apps.k8s.spark_submitter.spark_submit_arg_pb2 import SparkSubmitArg
from modules.tools.fuel_proxy.proto.job_config_pb2 import BosConfig, JobConfig

from modules.data.fuel.apps.lambda.web-portal.vehicle_calibration import VehicleCalibration

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.tpl')


@app.route('/submit_job', methods=['POST'])
def submit_job():

    # TODO(zongbao) Sanity Check
    if request.json['bos']:
        storage = {'bos': {
            'bucket': request.json['bucket'],
            'access_key': request.json['access_key'],
            'secret_key': request.json['access_secret']}
        }

    # TODO(zongbao) should be a global config
    PARTNERS = {
        'apollo',
        'apollo-evangelist',
        'apollo-qa',
        'udelv2019',
        'coolhigh',
    }

    if request.json['partner_id'] not in PARTNERS:
        msg = 'Sorry, you are not authorized to access this service!'
        return HTTPStatus.UNAUTHORIZED, msg

    job_config = {
        'partner_id': request.json['partner_id'],
        'job_type': request.json['job_type'],
        'input_data_path': request.json['input_data_path']
    }

    # Wait for service SIMPLE_HDMAP ready
    # if request.json['job_type'] == JobConfig.SIMPLE_HDMAP:
    #     job_config['zone_id'] = request.json['zone_id']
    #     job_config['lidar_type'] = request.json['lidar_type']

    job_id = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d-%H-%M')
    spark_submit_arg = SparkSubmitArg()
    partner = request.json['partner_id']
    spark_submit_arg.user.submitter = 'job_client'
    spark_submit_arg.user.running_role = partner
    spark_submit_arg.job.flags = f'--job_owner={partner} --job_id={job_id}'
    spark_submit_arg.partner.bos.bucket = storage['bucket']
    spark_submit_arg.partner.bos.access_key = storage['access_key']
    spark_submit_arg.partner.bos.secret_key = storage['secret_key']
    # Dispatch job.
    if self.job_config.job_type == JobConfig.VEHICLE_CALIBRATION:
        VehicleCalibration().submit(job_config, spark_submit_arg)
        msg = (f'Your job {job_id} is in process now! You will receive a notification in your '
               'corresponding email when it is finished.')
        return HTTPStatus.ACCEPTED, msg


def main(argv):
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    absl_app.run(main)
