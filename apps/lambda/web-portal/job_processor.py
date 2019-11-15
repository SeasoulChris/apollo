#!/usr/bin/env python

from http import HTTPStatus
import datetime
import os
import string
import time

from absl import logging

from modules.data.fuel.apps.k8s.spark_submitter.spark_submit_arg_pb2 import SparkSubmitArg
from modules.tools.fuel_proxy.proto.job_config_pb2 import BosConfig, JobConfig

from vehicle_calibration import VehicleCalibration


class JobProcessor(object):
    PARTNERS = {
        'apollo',
        'apollo-evangelist',
        'apollo-qa',
        'udelv2019',
        'coolhigh',
    }
    # BOS charsets.
    BOS_BUCKET_CHARSET = set(string.ascii_lowercase + string.digits + '-')
    BOS_KEY_CHARSET = set(string.hexdigits.lower())
    # Blob charsets.
    BLOB_ACCOUNT_CHARSET = set(string.ascii_lowercase + string.digits)
    BLOB_ACCESS_CHARSET = set(string.ascii_letters + string.digits + '+=/')  # Base64 encoding.
    BLOB_CONTAINER_CHARSET = set(string.ascii_lowercase + string.digits + '-')

    def __init__(self, job_config):
        self.job_config = job_config

    def process(self):
        # User authentication.
        partner = self.job_config.partner_id
        if partner not in self.PARTNERS:
            msg = 'Sorry, you are not authorized to access this service!'
            return HTTPStatus.UNAUTHORIZED, msg
        # QPS control.
        # TODO(xiaoxq): Use more elegant solution like sqlite.
        job_id = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M')
        mnt_path = os.path.join('/mnt', partner, job_id)
        if os.path.exists(mnt_path):
            msg = 'Please do not submit job too frequently! Only one job allowed per minute.'
            return HTTPStatus.TOO_MANY_REQUESTS, msg
        os.makedirs(mnt_path)

        ############################################################## New style
        # Construct arguments.
        spark_submit_arg = SparkSubmitArg()
        if not self.populate_storage_config(spark_submit_arg):
            return HTTPStatus.BAD_REQUEST, 'job_config format error!'
        spark_submit_arg.user.submitter = 'web_portal'
        spark_submit_arg.user.running_role = partner
        spark_submit_arg.job.flags = f'--job_owner={partner} --job_id={job_id}'
        # Dispatch job.
        if self.job_config.job_type == JobConfig.VEHICLE_CALIBRATION:
            VehicleCalibration().submit(self.job_config, spark_submit_arg)
            msg = (f'Your job {job_id} is in process now! You will receive a notification in your '
                   'corresponding email when it is finished.')
            return HTTPStatus.ACCEPTED, msg

        ############################################################## Old style
        # Construct arguments.
        bash_args = self.storage_config_to_cmd_args()
        if bash_args is None:
            return HTTPStatus.BAD_REQUEST, 'job_config format error!'
        py_args = f'--job_owner={partner} --job_id={job_id}'
        if self.job_config.job_type == JobConfig.SIMPLE_HDMAP:
            job_exec = 'bash generate_simple_hdmap.sh {} {} {}'.format(
                self.job_config.input_data_path,
                self.job_config.zone_id,
                self.job_config.lidar_type)
        elif self.job_config.job_type == JobConfig.PERCEPTION_MODEL_TRAINING:
            job_exec = 'bash perception_model_training.sh {} {}'.format(
                self.job_config.input_data_path,
                self.job_config.output_trained_model_path)
        else:
            return HTTPStatus.BAD_REQUEST, 'Unsupported job type.'

        cmd = 'nohup {} "{}" "{}" > /tmp/{}_{}.log 2>&1 &'.format(
            job_exec, bash_args, py_args, partner, job_id)
        logging.info(cmd)
        os.system(cmd)
        msg = ('Your job {} is in process now! You will receive a notification in your '
               'corresponding email when it is finished.'.format(job_id))
        return HTTPStatus.ACCEPTED, msg

    def storage_config_to_cmd_args(self):
        """
        Construct arguments from the given instance of
        modules.tools.fuel_proxy.proto.job_config_pb2.Storage.
        """
        storage = self.job_config.storage
        if storage.HasField('bos'):
            bos_conf = storage.bos
            # Bos config sanity check.
            if (set(bos_conf.bucket) > self.BOS_BUCKET_CHARSET or
                set(bos_conf.access_key) > self.BOS_KEY_CHARSET or
                set(bos_conf.secret_key) > self.BOS_KEY_CHARSET):
                return None
            return ('--partner_bos_region {} '
                    '--partner_bos_bucket {} '
                    '--partner_bos_access {} '
                    '--partner_bos_secret {} '.format(
                        BosConfig.Region.Name(bos_conf.region),
                        bos_conf.bucket,
                        bos_conf.access_key,
                        bos_conf.secret_key))
        elif storage.HasField('blob'):
            # Blob config sanity check.
            blob_conf = storage.blob
            if (set(blob_conf.storage_account) > self.BLOB_ACCOUNT_CHARSET or
                set(blob_conf.storage_access_key) > self.BLOB_ACCESS_CHARSET or
                set(blob_conf.blob_container) > self.BLOB_CONTAINER_CHARSET):
                return None
            return ('--azure_storage_account {} '
                    '--azure_storage_access_key {} '
                    '--azure_blob_container {}'.format(
                        blob_conf.storage_account,
                        blob_conf.storage_access_key,
                        blob_conf.blob_container))
        return None

    def populate_storage_config(self, spark_submit_arg):
        """
        Populate spark_submit_arg from modules.tools.fuel_proxy.proto.job_config_pb2.Storage.
        """
        storage = self.job_config.storage
        if storage.HasField('bos'):
            bos_conf = storage.bos
            # Bos config sanity check.
            if (set(bos_conf.bucket) > self.BOS_BUCKET_CHARSET or
                set(bos_conf.access_key) > self.BOS_KEY_CHARSET or
                set(bos_conf.secret_key) > self.BOS_KEY_CHARSET):
                return False
            spark_submit_arg.partner.bos.bucket = bos_conf.bucket
            spark_submit_arg.partner.bos.region = bos_conf.region
            spark_submit_arg.partner.bos.access_key = bos_conf.access_key
            spark_submit_arg.partner.bos.secret_key = bos_conf.secret_key
        elif storage.HasField('blob'):
            # Blob config sanity check.
            blob_conf = storage.blob
            if (set(blob_conf.storage_account) > self.BLOB_ACCOUNT_CHARSET or
                set(blob_conf.storage_access_key) > self.BLOB_ACCESS_CHARSET or
                set(blob_conf.blob_container) > self.BLOB_CONTAINER_CHARSET):
                return False
            spark_submit_arg.partner.blob.storage_account = blob_conf.storage_account
            spark_submit_arg.partner.blob.storage_access_key = blob_conf.storage_access_key
            spark_submit_arg.partner.blob.blob_container = blob_conf.blob_container
        return True
