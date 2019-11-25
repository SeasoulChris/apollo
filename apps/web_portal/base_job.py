#!/usr/bin/env python

from absl import logging
from google.protobuf.json_format import MessageToJson
import requests


class BaseJob(object):

    def submit(self, saas_job_arg, spark_submit_arg):
        """Submit job."""
        raise Exception('Not implemented!')

    @staticmethod
    def submit_one_job(spark_submit_arg):
        """Submit a job through spark_submitter service."""
        SUBMITTER = ('http://localhost:8001/api/v1/namespaces/default/services/'
                     'http:spark-submitter-service:8000/proxy/')
        res = requests.post(SUBMITTER, json=MessageToJson(spark_submit_arg))
        if res.ok:
            logging.info('Submit job successfully: {spark_submit_arg}')
        else:
            logging.error('Failed to submit job: {spark_submit_arg}')
