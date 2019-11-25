#!/usr/bin/env python

from base_job import BaseJob


class VehicleCalibration(BaseJob):

    def submit(self, job_arg, spark_submit_arg):
        """Submit job."""
        spark_submit_arg.job.entrypoint = 'fueling/control/calibration_table/vehicle_calibration.py'
        spark_submit_arg.job.flags += f' --input_data_path={job_arg.job.input_data_path}'
        spark_submit_arg.worker.count = 6
        spark_submit_arg.worker.cpu = 4
        spark_submit_arg.worker.memory = 60
        BaseJob.submit_one_job(spark_submit_arg)
