#!/usr/bin/env python

import os
import sys
import time

import torch

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
from fueling.planning.training_pipelines.trajectory_imitation_model_training_pipeline \
    import training


class PytorchTraining(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.train)
        logging.info('Training complete in {} seconds.'.format(
            time.time() - time_start))

    def train(self, instance_id):
        """Run training task"""

        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

        model_type = self.FLAGS.get('model_type')
        train_set_dir = self.FLAGS.get('train_set_dir')
        validation_set_dir = self.FLAGS.get('validation_set_dir')
        gpu_idx = self.FLAGS.get('gpu_idx')
        renderer_config_file = self.FLAGS.get('renderer_config_file')
        renderer_base_map_img_dir = self.FLAGS.get('renderer_base_map_img_dir')
        img_feature_rotation = self.FLAGS.get('img_feature_rotation')
        past_motion_dropout = self.FLAGS.get('past_motion_dropout')
        model_save_dir = self.FLAGS.get('model_save_dir')
        region = "sunnyvale_with_two_offices"
        renderer_base_map_data_dir = self.FLAGS.get(
            'renderer_base_map_data_dir')
        renderer_base_map_data_dir = "/apollo/modules/map/data/" + region + \
            "/base_map.bin" if renderer_base_map_data_dir is None else \
            renderer_base_map_data_dir

        training(model_type,
                 train_set_dir,
                 validation_set_dir,
                 gpu_idx,
                 renderer_config_file,
                 renderer_base_map_img_dir,
                 renderer_base_map_data_dir,
                 img_feature_rotation,
                 past_motion_dropout,
                 model_save_dir,
                 region)


if __name__ == '__main__':
    PytorchTraining().main()
