#!/usr/bin/env python

import os
import json
import signal
import subprocess
import threading
import time
import requests


import fueling.common.logging as logging
from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.learning.autotuner.cost_computation.job.base_cost_job import BaseCostJob

SERVICE_URL = "http://spark-submitter-service.default.svc.cluster.local:8000"

JOB_END_STATUS = {'Completed', 'Error', 'UnexpectedAdmissionError'}

WAIT_INTERVAL_SECONDS = 3


class K8sCostJob(BaseCostJob):
    def __init__(self):
        self.cancel_job = False

    def submit(self, options):
        entrypoint = "fueling/learning/autotuner/cost_computation/profiling_cost_computation.py"

        client_flags = {
            'node_selector': 'CPU',
            'workers': options.get("workers", 1),
            'role': options.get("role", ""),
            'cpu': 1,
            'gpu': 0,
            'memory': 5,  # in GB
            'disk': 5,  # in GB
        }

        client = SparkSubmitterClient(entrypoint, client_flags, options)
        spark_job_id = client.submit()
        cost_job_info = f"{options['role']}/{options['token']}/{options['iteration_id']}"
        self.wait(cost_job_info, spark_job_id)

    def wait(self, cost_job_info, spark_job_id):
        """Wait until the given job finishes"""
        self.cancel_job = False

        prev_job_status = None
        job_status = None
        is_up = False
        while job_status not in JOB_END_STATUS:
            if self.cancel_job:
                logging.warn(f"Cancelled spark job {spark_job_id} for {cost_job_info}.")
                return

            time.sleep(WAIT_INTERVAL_SECONDS)

            res = requests.get(SERVICE_URL, params={'job_id': spark_job_id})
            if not res.ok:
                logging.error('Failed to get job status.')
                continue

            job_status = json.loads(res.json() or '{}').get('status')
            if job_status == "Preparing" and is_up:
                """Preparing status happens when the pod is not found"""
                raise Exception(f"Job {spark_job_id} from {cost_job_info} is killed unexpectedly.")

            if job_status != "Preparing":
                is_up = True

            log_msg = f'{cost_job_info} with spark job {spark_job_id} is {job_status}...'
            if prev_job_status == job_status:
                logging.log_every_n_seconds(logging.INFO, log_msg, 90)
            else:
                logging.info(log_msg)

            prev_job_status = job_status

    def cancel(self):
        # TODO(vivian): delete the spark driver
        self.cancel_job = True
