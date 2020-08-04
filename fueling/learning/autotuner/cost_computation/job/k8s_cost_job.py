#!/usr/bin/env python

import json
import time
import requests


import fueling.common.logging as logging
from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.learning.autotuner.cost_computation.job.base_cost_job import BaseCostJob

SERVICE_URL = "http://spark-submitter-service.default.svc.cluster.local:8000"

JOB_END_STATUS = {'Completed', 'Error', 'UnexpectedAdmissionError', 'Terminating'}
JOB_SUCCESS_STATUS = {'Completed', 'Terminating'}
WAIT_INTERVAL_SECONDS = 3


class K8sCostJob(BaseCostJob):
    def __init__(self):
        self.cancel_job = False

    def run(self, options):
        entrypoint = "fueling/learning/autotuner/cost_computation/profiling_cost_computation.py"

        client_flags = {
            'workers': options.get("workers", 1),
            'role': options.get("role", ""),
            'cpu': 1,
            'gpu': 0,
            'memory': 5,  # in GB
            'disk': 5,  # in GB
        }

        client = SparkSubmitterClient(entrypoint, client_flags, options)
        cost_job_info = f"{options['role']}/{options['token']}/{options['iteration_id']}"
        spark_job_id = client.submit()
        if not spark_job_id:
            raise Exception(f"Failed to submit spark job: {cost_job_info}.")
        success = self.wait(cost_job_info, spark_job_id)
        return success

    def wait(self, cost_job_info, spark_job_id):
        """Wait until the given job finishes"""
        self.cancel_job = False

        prev_job_status = None
        job_status = None
        while job_status not in JOB_END_STATUS:
            if self.cancel_job:
                logging.warn(f"Cancelled spark job {spark_job_id} for {cost_job_info}.")
                return True

            time.sleep(WAIT_INTERVAL_SECONDS)

            res = requests.get(SERVICE_URL, params={'job_id': spark_job_id})
            if not res.ok:
                logging.error('Failed to get job status.')
                continue

            job_status = json.loads(res.json() or '{}').get('status')
            if (job_status == "Preparing" and prev_job_status is not None
                    and prev_job_status != job_status):
                # Preparing status happens when the pod is not found.
                # If not found for the second time, that means the job is completed.
                return True

            log_msg = f'{cost_job_info} with spark job {spark_job_id} is {job_status}...'
            if prev_job_status == job_status:
                logging.log_every_n_seconds(logging.INFO, log_msg, 90)
            else:
                logging.info(log_msg)

            prev_job_status = job_status

        logging.info(f'{cost_job_info} with spark job '
                     f'{spark_job_id} done with status {job_status}.')
        return job_status in JOB_SUCCESS_STATUS

    def cancel(self):
        # TODO(vivian): delete the spark driver
        self.cancel_job = True
