#!/usr/bin/env python

import os
import sys
import time

import torch

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
from fueling.planning.pipelines.trajectory_imitation_model_training_pipeline import traning


class PytorchTraining(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.train)
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))

    @staticmethod
    def train(instance_id):
        """Run training task"""

        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

        based_dir = '/mnt/bos/modules/planning/imitation/'
        model_type = 'rnn'
        train_dir = based_dir + 'traning_data/'
        valid_dir = based_dir + 'validation_data/'
        renderer_config_file \
            = based_dir + 'semantic_map_feature/planning_semantic_map_config.pb.txt'

        imgs_dir = based_dir + "imitation/semantic_map_features/"
        input_data_augmentation = False
        model_dir = based_dir + "imitation/model/"

        traning(model_type, train_dir, valid_dir, renderer_config_file,
                imgs_dir, input_data_augmentation, model_dir)


if __name__ == '__main__':
    PytorchTraining().main()
