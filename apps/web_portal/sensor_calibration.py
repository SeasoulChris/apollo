#!/usr/bin/env python

from base_job import BaseJob


class SensorCalibration(BaseJob):

    def submit(self, job_arg, spark_submit_arg):
        """Submit job."""
        spark_submit_arg.job.entrypoint = \
            'fueling/perception/sensor_calibration/calibration_multi_sensors.py'
        spark_submit_arg.job.flags += f' --input_data_path={job_arg.flags.get("input_data_path")}'
        spark_submit_arg.worker.count = 2
        spark_submit_arg.worker.cpu = 1
        spark_submit_arg.worker.memory = 24
        BaseJob.submit_one_job(spark_submit_arg)
