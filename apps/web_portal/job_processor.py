#!/usr/bin/env python

from http import HTTPStatus
import datetime
import os
import string
import time

from absl import logging

from fueling.common.partners import partners
from modules.data.fuel.apps.k8s.spark_submitter.spark_submit_arg_pb2 import SparkSubmitArg
from modules.data.fuel.apps.web_portal.saas_job_arg_pb2 import SaasJob

from control_profiling import ControlProfilingMetrics
from sensor_calibration import SensorCalibration
from vehicle_calibration import VehicleCalibration


class JobProcessor(object):
    # BOS charsets.
    BOS_KEY_CHARSET = set(string.hexdigits.lower())
    # Blob charsets.
    BLOB_ACCOUNT_CHARSET = set(string.ascii_lowercase + string.digits)
    BLOB_ACCESS_CHARSET = set(string.ascii_letters + string.digits + '+=/')  # Base64 encoding.

    JOB_PROCESSORS = {
        SaasJob.CONTROL_PROFILING: ControlProfilingMetrics,
        SaasJob.SENSOR_CALIBRATION: SensorCalibration,
        SaasJob.VEHICLE_CALIBRATION: VehicleCalibration,
    }

    def __init__(self, job_arg):
        self.job_arg = job_arg
        self.partner_account = partners.get(job_arg.partner.id)

    def process(self):
        # User authentication.
        if self.partner_account is None:
            msg = 'Sorry, you are not authorized to access this service!'
            return HTTPStatus.UNAUTHORIZED, msg
        # Construct arguments.
        spark_submit_arg = SparkSubmitArg()
        if not self.populate_storage_config(spark_submit_arg):
            return HTTPStatus.BAD_REQUEST, 'job_arg format error!'
        spark_submit_arg.user.submitter = self.job_arg.partner.id
        # Dispatch job.
        processor = self.JOB_PROCESSORS.get(self.job_arg.job.job_type)
        if processor is None:
            return HTTPStatus.BAD_REQUEST, 'Unsupported job type.'
        processor().submit(self.job_arg, spark_submit_arg)
        return (HTTPStatus.ACCEPTED, 'Your job is in process now! You will receive a '
                'notification in your corresponding email when it is finished.')


    def populate_storage_config(self, spark_submit_arg):
        """Populate spark_submit_arg from saas_job_arg and stored partner_account."""
        if self.job_arg.partner.bos.access_key:
            bos = self.job_arg.partner.bos
            # Bos config sanity check.
            if not self.partner_account.bos_bucket:
                logging.error('User requested to use BOS while does not have it on profile.')
                return False
            if (set(bos.access_key) > self.BOS_KEY_CHARSET or
                    set(bos.secret_key) > self.BOS_KEY_CHARSET):
                logging.error('User provided invalid information.')
                return False
            spark_submit_arg.partner.bos.bucket = self.partner_account.bos_bucket
            spark_submit_arg.partner.bos.region = self.partner_account.bos_region
            spark_submit_arg.partner.bos.access_key = bos.access_key
            spark_submit_arg.partner.bos.secret_key = bos.secret_key
        elif self.job_arg.partner.blob.storage_account:
            # Blob config sanity check.
            blob = self.job_arg.partner.blob
            if (set(blob.storage_account) > self.BLOB_ACCOUNT_CHARSET or
                set(blob.storage_access_key) > self.BLOB_ACCESS_CHARSET):
                return False
            spark_submit_arg.partner.blob.storage_account = blob.storage_account
            spark_submit_arg.partner.blob.storage_access_key = blob.storage_access_key
            spark_submit_arg.partner.blob.blob_container = self.partner_account.blob_container
        return True
