#!/usr/bin/env python

from base_job import BaseJob


class SimpleHDMap(BaseJob):

    def submit(self, job_arg, spark_submit_arg):
        """Submit job."""
        spark_submit_arg.job.entrypoint = 'fueling/map/generate_maps.py'
        spark_submit_arg.job.flags += (
            f' --input_data_path={job_arg.flags.get("input_data_path")}'
            f' --output_data_path={job_arg.flags.get("output_data_path")}'
            f' --zone_id={job_arg.flags.get("zone_id")}'
            f' --lidar_type={job_arg.flags.get("lidar_type")}')
        spark_submit_arg.worker.count = 2
        spark_submit_arg.worker.cpu = 1
        spark_submit_arg.worker.memory = 24
        spark_submit_arg.partner.storage_writable = True
        BaseJob.submit_one_job(spark_submit_arg)
