#!/usr/bin/env python

from http import HTTPStatus
import datetime
import json
import os
import string
import time

from modules.tools.fuel_proxy.proto.job_config_pb2 import BosConfig


class VehicleCalibration(object):
    """Vehicle Calibration restful service"""
    BOS_BUCKET_CHARSET = set(string.ascii_lowercase + string.digits + '-')
    BOS_KEY_CHARSET = set(string.hexdigits.lower())

    @staticmethod
    def process(job_config):
        """Accept user request, verify and process."""
        job_id = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M')
        mnt_path = os.path.join('/mnt', job_config.partner_id, job_id)
        if os.path.exists(mnt_path):
            msg = 'Please do not submit job too frequently! Only one job allowed per minute.'
            return json.dumps({'message': msg}), HTTPStatus.TOO_MANY_REQUESTS
        os.makedirs(mnt_path)

        # Bos config sanity check.
        bos_config = job_config.storage.bos
        if set(bos_config.bucket) > VehicleCalibration.BOS_BUCKET_CHARSET:
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST
        if set(bos_config.access_key) > VehicleCalibration.BOS_KEY_CHARSET:
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST
        if set(bos_config.secret_key) > VehicleCalibration.BOS_KEY_CHARSET:
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST

        # Job summit: vehicle_calibration.sh <bash args> <python args> <input_data_path>
        os.system('nohup bash vehicle_calibration.sh \
            "--partner_bos_region {} --partner_bos_bucket {} \
            --partner_bos_access {} --partner_bos_secret {}" \
            "--job_owner={} --job_id={}" "{}" > /tmp/{}_{}.log 2>&1 &'.format(
                BosConfig.Region.Name(bos_config.region), bos_config.bucket,
                bos_config.access_key, bos_config.secret_key,
                job_config.partner_id, job_id,
                job_config.input_data_path,
                job_config.partner_id, job_id))

        msg = ('Your job {} is in process now! You will receive a notification in your '
               'corresponding email when it is finished.'.format(job_id))
        return json.dumps({'message': msg}), HTTPStatus.ACCEPTED
