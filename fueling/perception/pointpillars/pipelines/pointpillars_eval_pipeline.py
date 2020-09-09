#!/usr/bin/env python

import os
import sys
import time
import torch
import cv2 as cv

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
from fueling.perception.pointpillars.second.pytorch.train import evaluate


class PointPillarsEvaluate(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.eval)
        logging.info('Evaluate complete in {} seconds.'.format(time.time() - time_start))

    def eval(self, instance_id):
        """Run Evaluate task"""

        cv.setNumThreads(0)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

        config_path = '/fuel/fueling/perception/pointpillars/second/'\
                      'configs/all.pp.mhead.cloud.config'
        model_dir = '/fuel/fueling/perception/pointpillars/second/models'
        ckpt_path = '/fuel/fueling/perception/pointpillars/second/models/voxelnet-5865.tckpt'
        '''
        config_path = '/mnt/bos/modules/perception/pointpillars/config/all.pp.mhead.config'
        model_dir = '/mnt/bos/modules/perception/pointpillars/models/'
        ckpt_path = '/mnt/bos/modules/perception/pointpillars/models/voxelnet-5865.tckpt'
        '''
        evaluate(config_path, model_dir, ckpt_path=ckpt_path, measure_time=True, batch_size=1)


if __name__ == '__main__':
    PointPillarsEvaluate().main()
