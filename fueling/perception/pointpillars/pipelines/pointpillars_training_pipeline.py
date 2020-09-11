#!/usr/bin/env python

import cv2 as cv
import os
import sys
import time
import torch

from fueling.common.base_pipeline import BasePipeline
from fueling.common.job_utils import JobUtils
from fueling.perception.pointpillars.second.pytorch.train import train
import fueling.common.context_utils as context_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


class PointPillarsTraining(BasePipeline):
    """Demo pipeline."""

    def run_test(self):

        config_path = '/fuel/testdata/perception/pointpillars/' \
                      'all.pp.mhead.cloud.config'
        model_dir = '/fuel/testdata/perception/pointpillars/models/'
        input_data_path = '/fuel/testdata/perception/pointpillars/kitti_testdata'
        train(config_path, model_dir, input_data_path, unit_test=True)

    def run(self):
        """Run."""
        self.if_error = False
        job_id = self.FLAGS.get('job_id')
        output_data_path = self.FLAGS.get('output_data_path')
        input_data_path = self.FLAGS.get('input_data_path')
        object_storage = self.partner_storage() or self.our_storage()
        self.output_data_path = object_storage.abs_path(output_data_path)
        self.input_data_path = object_storage.abs_path(input_data_path)

        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.training)
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_progress(90)
            if self.if_error:
                JobUtils(job_id).save_job_failure_code('E702')

    def training(self, instance_id):
        """Run training task"""
        cv.setNumThreads(0)

        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

        config_path = file_utils.fuel_path(
            'testdata/perception/pointpillars/'
            'all.pp.mhead.cloud.config')
        model_dir = os.path.join(self.output_data_path, 'models/')
        pretrained_path = file_utils.fuel_path(
            'testdata/perception/pointpillars/voxelnet-nuscenes-58650.tckpt')

        try:
            train(config_path, model_dir, self.input_data_path, pretrained_path=pretrained_path)
        except BaseException:
            logging.error('Failed to training model')
            self.if_error = True


if __name__ == '__main__':
    PointPillarsTraining().main()
