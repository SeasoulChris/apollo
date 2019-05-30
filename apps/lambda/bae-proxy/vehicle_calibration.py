#!/usr/bin/env python

from http import HTTPStatus
import json
import os

import fueling.common.time_utils as time_utils


class VehicleCalibration(object):
    """Vehicle Calibration restful service"""

    @staticmethod
    def process(job_config):
        """Accept user request, verify and process."""
        job_id = time_utils.format_current_time(
            '%Y-%m-%d-%H-%M-%S' if job_config.partner_id == 'apollo' else '%Y-%m-%d-%H-%M')
        mnt_path = os.path.join('/mnt', job_config.partner_id, job_id)
        if os.path.exists(mnt_path):
            msg = 'Please do not submit job too frequently!'
            return json.dumps({'message': msg}), HTTPStatus.TOO_MANY_REQUESTS
        os.makedirs(mnt_path)

        # Bos config sanity check.
        bos_config = job_config.storage.bos
        if set(bos_config.bucket).difference(set('0123456789abcdefghijklmnopqrstuvwxyz-')):
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST
        if set(bos_config.endpoint).difference(set('0123456789abcdefghijklmnopqrstuvwxyz-_:/.')):
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST
        if set(bos_config.access_key).difference(set('0123456789abcdef')):
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST
        if set(bos_config.secret_key).difference(set('0123456789abcdef')):
            return json.dumps({'message': 'job_config format error!'}), HTTPStatus.BAD_REQUEST

        # Job summit.
        os.system('cd /apollo/modules/data/fuel && '
                  'nohup bash tools/submit-job-to-k8s.sh -w 1 -c 1 -m 1g -d 1 '
                  '--partner_bos_bucket "{}" --partner_bos_endpoint "{}" '
                  '--partner_bos_access "{}" --partner_bos_secret "{}" '
                  'fueling/demo/count-msg-by-channel.py '
                  '--job_owner="{}" --job_id="{}" --input_data_path="{}" &'.format(
                      bos_config.bucket, bos_config.endpoint,
                      bos_config.access_key, bos_config.secret_key,
                      job_config.partner_id, job_id, job_config.input_data_path))

        msg = ('Your job {} is in process now! You will receive a notification in your '
               'corresponding email when it is finished.'.format(task_id))
        return json.dumps({'message': msg}), HTTPStatus.ACCEPTED
