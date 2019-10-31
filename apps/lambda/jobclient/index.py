#!/usr/bin/env python

import json

import flask
from flask import request
from absl import flags
from absl import logging
from absl import app as absl_app
import requests

from modules.tools.fuel_proxy.proto.job_config_pb2 import BosConfig, JobConfig

app = flask.Flask(__name__)

flags.DEFINE_string('fuel_proxy', 'https://apollofuel0.bceapp.com:8443',
                    'Endpoint of Apollo-Fuel proxy.')


@app.route('/')
def index():
    return flask.render_template('index.tpl')


@app.route('/submit_job', methods=['POST'])
def submit_job():

    if request.json['bos']:
        storage = {'bos': {
            'bucket': request.json['bucket'],
            'access_key': request.json['access_key'],
            'secret_key': request.json['access_secret']}
        }
    elif request.json['blob']:
        storage = {'blob': {
            'blob_container': request.json['bucket'],
            'storage_account': request.json['access_key'],
            'storage_account_key': request.json['access_secret']}
        }

    partner_storage_writable = request.json['partner_storage_writable']

    request_dict = {
        'partner_id': request.json['partner_id'],
        'job_type': request.json['job_type'],
        'input_data_path': request.json['input_data_path'],
        'storage': storage
        # 'partner_storage_writable': partner_storage_writable
    }

    if request.json['job_type'] == JobConfig.SIMPLE_HDMAP:
        request_dict['zone_id'] = request.json['zone_id']
        request_dict['lidar_type'] = request.json['lidar_type']

    request_json = json.dumps(request_dict)
    request_post = requests.post(flags.FLAGS.fuel_proxy, json=request_json,
                                 verify='../bae-proxy/ssl_keys/cert.pem')
    response = json.loads(request_post.json()) if request_post.json() else {}

    if request_post.ok:
        logging.info(response.get('message') or 'OK')
    else:
        logging.error(
            response.get('message') or
            'Request failed with HTTP code {}'.format(request_post.status_code))

    return response


def main(argv):
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    absl_app.run(main)
