#!/usr/bin/env python3

import base64
import getpass
import json
import os
import pprint
import sys
import time

from absl import flags
from absl import logging
import requests

import fueling.common.file_utils as file_utils


# User.
flags.DEFINE_string('role', 'apollo', 'Running as another role instead of the job submitter.')

# Env.
flags.DEFINE_string('image', 'hub.baidubce.com/apollo/spark:bazel2', 'Docker image.')
flags.DEFINE_enum('node_selector', 'CPU', ['CPU', 'GPU', 'ANY'], 'Node selector.')
flags.DEFINE_enum('log_verbosity', 'INFO', ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
                  'Log verbosity.')

# Job.
flags.DEFINE_boolean('cloud', False, 'Run pipeline in cloud.')
flags.DEFINE_boolean('wait', False, 'Whether to wait to finish.')

# Worker.
flags.DEFINE_integer('workers', 1, 'Worker count.', short_name='w')
flags.DEFINE_integer('cpu', 1, 'Worker CPU cores.', short_name='c')
flags.DEFINE_integer('gpu', 0, 'Worker GPU cores.', short_name='g')
flags.DEFINE_integer('memory', 12, 'Worker memory in GB.', short_name='m')
flags.DEFINE_integer('disk', 20, 'Worker disk in GB.', short_name='d')

# Partner.
flags.DEFINE_boolean('partner_storage_writable', False, 'Mount partner storage as writable.')
flags.DEFINE_string('partner_bos_bucket', None, 'Partner bos bucket.')
flags.DEFINE_string('partner_bos_region', None, 'Partner bos region.')
flags.DEFINE_string('partner_bos_access', None, 'Partner bos access.')
flags.DEFINE_string('partner_bos_secret', None, 'Partner bos secret.')

# Internal use.
flags.DEFINE_string('kube_proxy', 'usa-data.baidu.com', 'Kube proxy.')
flags.DEFINE_string('spark_submitter_service_url', None, 'URL of the Spark Submitter service')


class SparkSubmitterClient(object):
    def __init__(self, entrypoint, client_flags={}, job_flags=None):
        self.zip_app = self.entrypoint_to_zip_app(entrypoint)
        self.client_flags = client_flags
        self.job_flags = job_flags if job_flags is not None else self.collect_job_flags()
        logging.info(F'job_flags collected as: {self.job_flags}')
        logging.info(F'Submitting zip_app {self.zip_app} for entrypoint {entrypoint}')

    def submit(self):
        """Tool entrypoint."""
        # Construct argument according to apps/k8s/spark_submitter/spark_submit_arg.proto
        arg = {
            'user': self.get_user(),
            'env': self.get_env(),
            'job': self.get_job(),
            'worker': self.get_worker(),
            'partner': self.get_partner(),
        }

        # Submit job.
        service_url = self.client_flags.get('spark_submitter_service_url') or \
        flags.FLAGS.spark_submitter_service_url or self.get_service_url()

        res = requests.post(service_url, json=json.dumps(arg))
        payload = json.loads(res.json() or '{}')

        arg['job'].pop('fueling_zip_base64')
        logging.info('SparkSubmitArg is')
        pprint.PrettyPrinter(indent=2).pprint(arg)

        # Process result.
        if not res.ok:
            logging.error('Failed to submit job: HTTP {}, {}'.format(
                res.status_code, payload.get('error')))
            return

        job_id = payload.get('job_id')
        logging.info('Job {} submitted!'.format(job_id))
        # TODO: logging.info('View your task at ...')

        # Wait until job finishes.
        if not self.client_flags.get('wait', self.get_default('wait')):
            return
        WAIT_INTERVAL_SECONDS = 3
        END_STATUS = {'Completed', 'Error', 'UnexpectedAdmissionError'}
        job_status = None
        while job_status not in END_STATUS:
            time.sleep(WAIT_INTERVAL_SECONDS)
            res = requests.get(service_url, params={'job_id': job_id})
            if res.ok:
                job_status = json.loads(res.json() or '{}').get('status')
                logging.info('Job is {}...'.format(job_status))
            else:
                logging.error('Failed to get job status.')
        if job_status != 'Completed':
            sys.exit(1)

    def get_service_url(self):
        kube_proxy = flags.FLAGS.kube_proxy
        kube_proxy_url = F'http://{kube_proxy}:8001'
        service_name = 'http:spark-submitter-service:8000'
        service_url = F'{kube_proxy_url}/api/v1/namespaces/default/services/{service_name}/proxy/'

        if kube_proxy != 'localhost' and os.system(F'ping -c 1 {kube_proxy}') != 0:
            logging.fatal(F'Cannot reach {kube_proxy}. Are you running in intranet?')
            sys.exit(1)

        return service_url

    def get_user(self):
        return {
            'submitter': getpass.getuser(),
            'running_role': self.client_flags.get('role', self.get_default('role')),
        }

    def get_env(self):
        return {
            'docker_image': self.client_flags.get('image', self.get_default('image')),
            'node_selector': self.client_flags.get('node_selector', self.get_default('node_selector')),
            'log_verbosity': self.client_flags.get('log_verbosity', self.get_default('log_verbosity')),
        }

    def get_job(self):
        with open(self.zip_app, "rb") as fin:
            encoded_zip = base64.b64encode(fin.read()).decode('ascii')
        logging.info('Job has %.2fMB.' % (len(encoded_zip) / (2**20)))

        return {
            'entrypoint': self.zip_app,
            'fueling_zip_base64': encoded_zip,
            'flags': ' '.join([F'--{name}={value}' for name, value in self.job_flags.items()]),
        }

    def get_worker(self):
        return {
            'count': self.client_flags.get('workers', self.get_default('workers')),
            # cpu/gpu could be specifically set to 0
            'cpu': self.client_flags.get('cpu', self.get_default('cpu')),
            'gpu': self.client_flags.get('gpu', self.get_default('gpu')),
            'memory': self.client_flags.get('memory', self.get_default('memory')),
            'disk': self.client_flags.get('disk', self.get_default('disk')),
        }

    def get_partner(self):
        # partner_storage_writable could be specifically set to False
        partner = {'storage_writable': self.client_flags.get('partner_storage_writable', \
            self.get_default('partner_storage_writable'))}

        # partner_bos_bucket could be specifically set to None
        if self.client_flags.get('partner_bos_bucket', self.get_default('partner_bos_bucket')):
            partner['bos'] = {
                'bucket': self.client_flags.get('partner_bos_bucket', self.get_default('partner_bos_bucket')),
                'access_key': self.client_flags.get('partner_bos_access', self.get_default('partner_bos_access')),
                'secret_key': self.client_flags.get('partner_bos_secret', self.get_default('partner_bos_secret')),
                'region': self.client_flags.get('partner_bos_region', self.get_default('partner_bos_region')),
            }
        return partner

    """
    Use this function instead of directly accessing flags to avoid instant parsing of flags and UnparsedFlagAccessError.
    e.g dict.get(key[, default]) will parse flags even if not needed when using flags as default
    """
    def get_default(self, flag):
        return flags.FLAGS[flag].value

    @staticmethod
    def entrypoint_to_zip_app(entrypoint):
        if not entrypoint.endswith('.py'):
            logging.fatal(F'Cannot process entrypoint {entrypoint}')

        entrypoint = file_utils.fuel_path(entrypoint)
        zip_app = entrypoint[:-3] + '.zip'
        if not os.path.exists(zip_app):
            FUEL_DIR = '/fuel/'
            fuel_pos = zip_app.find(FUEL_DIR)
            if fuel_pos < 0:
                logging.fatal('Pipeline should be under fuel/ dir.')
            zip_app = os.path.join('/fuel/bazel-bin', zip_app[fuel_pos + len(FUEL_DIR):])
        return zip_app

    @staticmethod
    def collect_job_flags():
        IGNORE_MODULES = {
            'absl.logging',
            'apps.k8s.spark_submitter.client',
        }
        job_flags = {}
        for module, module_flags in flags.FLAGS.flags_by_module_dict().items():
            if module in IGNORE_MODULES:
                continue
            for flag in module_flags:
                if not flag.using_default_value:
                    job_flags[flag.name] = flag.value
        return job_flags
