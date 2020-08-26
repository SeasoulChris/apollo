#!/usr/bin/env python

import os
import sys
import time
from absl import flags
import torch
import cv2 as cv

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
from fueling.perception.pointpillars.second.pytorch.train import train

flags.DEFINE_string('config_path', None, 'training config file')
flags.DEFINE_string('pretrained_path', None, 'finetune pertrained model path')
flags.DEFINE_string('model_dir', None, 'training models saved dir')


class PointPillarsTraining(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.training)
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))

    def training(self, instance_id):
        """Run training task"""
        cv.setNumThreads(0)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

        config_path = self.FLAGS.get('config_path')
        model_dir = self.FLAGS.get('model_dir')
        pretrained_path = self.FLAGS.get('pretrained_path')

        train(config_path, model_dir, pretrained_path=pretrained_path)


if __name__ == '__main__':
    PointPillarsTraining().main()
