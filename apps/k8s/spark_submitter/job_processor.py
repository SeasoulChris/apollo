#!/usr/bin/env python

from http import HTTPStatus
import string

from absl import logging

from apps.k8s.spark_submitter.saas_job_arg_pb2 import SaasJobArg
import apps.k8s.spark_submitter.jobs as jobs


class JobProcessor(object):
    # BOS charsets.
    BOS_KEY_CHARSET = set(string.hexdigits.lower())
    # Flag value charsets.
    DISALLOWED_FLAG_VALUE_CHARSET = set('&|;\n\r')

    JOB_PROCESSORS = {
        SaasJobArg.CONTROL_PROFILING: jobs.ControlProfiling,
        SaasJobArg.OPEN_SPACE_PLANNER_PROFILING: jobs.OpenSpacePlannerProfiling,
        SaasJobArg.SENSOR_CALIBRATION: jobs.SensorCalibration,
        SaasJobArg.VEHICLE_CALIBRATION: jobs.VehicleCalibration,
        SaasJobArg.VIRTUAL_LANE_GENERATION: jobs.VirtualLaneGeneration,
    }

    def __init__(self, job_arg):
        self.job_arg = job_arg

    def process(self):
        # Check flag values.
        for flag_value in self.job_arg.flags.values():
            if set(flag_value).intersection(self.DISALLOWED_FLAG_VALUE_CHARSET):
                msg = 'Please check your input.'
                logging.error(msg)
                return HTTPStatus.BAD_REQUEST, msg
        processor = self.JOB_PROCESSORS.get(self.job_arg.job_type)
        if processor is None:
            msg = 'Unsupported job type.'
            logging.error(msg)
            return HTTPStatus.BAD_REQUEST, msg
        # Construct client_flags.
        client_flags = {
            'role': 'OPEN_SERVICES',
            'node_selector': 'FOR_OPEN_SERVICES',
            'partner_job_type': SaasJobArg.JobType.Name(self.job_arg.job_type).lower(),
        }
        vehicle_sn = self.job_arg.partner.vehicle_sn
        if vehicle_sn:
            client_flags['partner_vehicle_sn'] = vehicle_sn
        partner_email = self.job_arg.partner.email
        if partner_email:
            client_flags['partner_email'] = partner_email
        if not self.populate_storage_config(client_flags):
            msg = 'job_arg format error!'
            logging.error(msg)
            return HTTPStatus.BAD_REQUEST, msg
        job_flags = {}
        processor().submit(self.job_arg, client_flags, job_flags)
        msg = ('Your job is in process now! You will receive a '
               'notification in your corresponding email when it is finished.')
        logging.info(msg)
        return HTTPStatus.ACCEPTED, msg

    def populate_storage_config(self, client_flags):
        """Check flags from saas_job_arg and stored partner_account."""
        if self.job_arg.partner.bos.access_key:
            bos = self.job_arg.partner.bos
            # Bos config sanity check.
            if not bos.bucket:
                logging.error('User requested to use BOS while does not indicate the bucket.')
                return False
            if (set(bos.access_key) > self.BOS_KEY_CHARSET
                    or set(bos.secret_key) > self.BOS_KEY_CHARSET):
                logging.error('User provided invalid information.')
                return False
            client_flags.update({
                'partner_bos_bucket': bos.bucket,
                'partner_bos_region': bos.region,
                'partner_bos_access': bos.access_key,
                'partner_bos_secret': bos.secret_key,
            })
            return True
        return False
