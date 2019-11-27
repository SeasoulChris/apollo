#!/usr/bin/env python

from base_job import BaseJob
from modules.data.fuel.apps.k8s.spark_submitter.spark_submit_arg_pb2 import Env


class PerceptionModelTraining(BaseJob):

    def submit(self, job_arg, spark_submit_arg):
        """Submit job."""
        spark_submit_arg.job.entrypoint = 'fueling/perception/YOLOv3/yolov3_training.py'
        spark_submit_arg.job.flags += (
            f' --input_data_path={job_arg.flags.get("input_data_path")}'
            f' --output_data_path={job_arg.flags.get("output_data_path")}')
        spark_submit_arg.worker.count = 1
        spark_submit_arg.worker.cpu = 1
        spark_submit_arg.worker.memory = 20
        spark_submit_arg.partner.storage_writable = True
        spark_submit_arg.env.node_selector = Env.NodeSelector.GPU
        BaseJob.submit_one_job(spark_submit_arg)
