#!/usr/bin/env python

import os
import subprocess
import signal

from fueling.learning.autotuner.cost_computation.job.base_cost_job import BaseCostJob
import fueling.common.logging as logging


class LocalCostJob(BaseCostJob):
    def __init__(self):
        self.process = None

    def submit(self, options):
        if self.process:
            logging.error('A job is running')
            return False

        job_cmd = "bazel run //fueling/learning/autotuner/cost_computation:profiling_cost_computation"
        option_strings = [f"--{name}={value}" for (name, value) in options.items()]
        cmd = f"cd /fuel; {job_cmd} -- {' '.join(option_strings)}"
        logging.info(f"Executing '{cmd}'")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            shell=True,
            preexec_fn=os.setsid)
        try:
            self.process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            logging.error(f'Time out. Killing cost job {self.process.pid}')
            self.cancel()
            return False
        else:
            self.process = None
            return True

    def cancel(self):
        if not self.process:
            return

        logging.info(f'Cancelling cost job {self.process.pid}')
        # Send the signal to all the process groups
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        self.process = None
