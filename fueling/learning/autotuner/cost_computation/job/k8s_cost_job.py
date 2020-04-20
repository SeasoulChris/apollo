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
        entrypoint = "fueling/learning/autotuner/cost_computation/control_cost_computation.py"
        self.client = SparkSubmitterClient(entrypoint, {}, options)
        self.client.submit()

    def cancel(self):
        # Not support yet
        self.client = None
        return
