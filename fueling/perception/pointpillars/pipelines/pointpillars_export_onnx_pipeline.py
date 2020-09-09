#!/usr/bin/env python

import os
import time

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.pointpillars.second.pytorch.trans_onnx import trans_onnx
import fueling.common.logging as logging
from fueling.common.job_utils import JobUtils
import fueling.common.context_utils as context_utils
import fueling.common.file_utils as file_utils
import fueling.common.email_utils as email_utils


class PointPillarsExportOnnx(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        job_id = self.FLAGS.get('job_id')
        output_data_path = self.FLAGS.get('output_data_path')
        object_storage = self.partner_storage() or self.our_storage()
        self.output_data_path = object_storage.abs_path(output_data_path)

        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.export_onnx)
        logging.info(
            'pointpillars export onnx complete in {} seconds.'.format(
                time.time() - time_start))
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_progress(100)
            self.send_email_notification(output_data_path)

    def export_onnx(self, instance_id):
        """Run export onnx task"""

        config_path = file_utils.fuel_path(
            'testdata/perception/pointpillars/'
            'all.pp.mhead.cloud.config')

        model_path = os.path.join(self.output_data_path, 'models/voxelnet-5865.tckpt')
        save_onnx_path = self.output_data_path

        trans_onnx(config_path, model_path, save_onnx_path)

    def send_email_notification(self, model_path):
        """Send email notification to users"""
        title = 'Your perception lidar model training job is done!'
        content = {
            'Job Owner': self.FLAGS.get('job_owner'),
            'Job ID': self.FLAGS.get('job_id'),
            'Model Path': model_path,
        }
        receivers = email_utils.PREDICTION_TEAM
        if os.environ.get('PARTNER_EMAIL'):
            receivers.append(os.environ.get('PARTNER_EMAIL'))
        email_utils.send_email_info(title, content, receivers)


if __name__ == '__main__':
    PointPillarsExportOnnx().main()
