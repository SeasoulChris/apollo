#!/usr/bin/env python

from http import HTTPStatus
import json
import os

import fueling.common.time_utils as time_utils


class VehicleCalibration(object):
    """Vehicle Calibration restful service"""

    def __init__(self, job_config):
        self.job_config = job_config

    def process(self):
        """Accept user request, verify and process."""
        task_id_fmt = '%Y-%m-%d-%H-%M-%S' if job_config.partner_id == 'apollo' else '%Y-%m-%d-%H-%M'
        task_id = time_utils.format_current_time(task_id_fmt)
        mnt_path = os.path.join('/mnt', job_config.partner_id, task_id)
        if os.path.exists(mnt_path):
            msg = 'Please do not submit job too frequently!'
            return json.dumps({'message': msg}), HTTPStatus.TOO_MANY_REQUESTS
        os.makedirs(mnt_path)

        # TODO: Bos config sanity check.
        bos_config = self.job_config.storage.bos
        # bos_config.bucket
        # bos_config.endpoint
        # bos_config.access_key
        # bos_config.secret_key

        # TODO: Job summit.
        # bos_config.4
        # job_config.partner_id
        # task_id
        os.system('cd /apollo/modules/data/fuel && '
                  'nohup bash tools/submit-job-to-k8s.sh -w 1 -c 1 -m 1g -d 1 '
                  'fueling/demo/count-msg-by-channel.py &')

        msg = ('Your job {} is in process now! You will receive a notification in your '
               'corresponding email when it is finished.'.format(task_id))
        return json.dumps({'message': msg}), HTTPStatus.ACCEPTED
