#!/usr/bin/env python

import os
import subprocess
import threading
import signal

from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.learning.autotuner.cost_computation.job.base_cost_job import BaseCostJob


class K8sCostJob(BaseCostJob):
    def __init__(self):
        self.client = None

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
            'wait': True,
        }

        self.client = SparkSubmitterClient(entrypoint, client_flags, options)
        self.client.submit()

    def cancel(self):
        # Not support yet
        self.client = None
        return
