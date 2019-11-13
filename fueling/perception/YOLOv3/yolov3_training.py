#!/usr/bin/env python3

import os
import shutil

import glob
import numpy as np

from absl import flags
from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.train import training
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client
import fueling.perception.YOLOv3.utils.data_utils as data_utils


flags.DEFINE_string('input_training_data_path', '', 'Input data path for training.')
flags.DEFINE_string('output_trained_model_path', '', 'Output path for trained model.')


class Yolov3Training(BasePipeline):
    """Model training pipeline."""
    
    def run_test(self):
        """Run test."""
        training_datasets = glob.glob(os.path.join(cfg.train_data_dir_local, '*'))
        self.run(training_datasets)

    def run_prod(self):
        """Run prod."""
        input_data_path = self.FLAGS.get('input_training_data_path') or cfg.train_data_dir_cloud
        object_storage = self.partner_object_storage() or BosClient()

        self.run([object_storage.abs_path(input_data_path)])

    def run(self, training_datasets):
        """Run the actual pipeline job."""

        def _executor(image_paths, output_trained_model_path):
            """Executor task that runs on workers"""
            if not image_paths:
                logging.warn('no images found in this set')
                return

            logging.info('current image set size: {}'.format(len(image_paths)))

            engine = training()
            engine.setup_training(output_trained_model_path)
            data_pool = Dataset(image_paths)

            for _ in range(cfg.max_iter):
                data_batch = data_pool.batch
                engine.step(data_batch, output_trained_model_path)


        logging.info('input training data path: {}'.format(training_datasets))
        
        config_path = '/apollo/modules/data/fuel/fueling/perception/YOLOv3/config.py'
        model_output_path = self.FLAGS.get('output_trained_model_path') or cfg.model_output_path
        object_storage = self.partner_object_storage() or BosClient()
        model_output_path = object_storage.abs_path(model_output_path)

        logging.info('output trained model path: {}'.format(model_output_path))

        file_utils.makedirs(model_output_path)
        shutil.copyfile(config_path, os.path.join(model_output_path, 'config.py'))

        training_datasets_rdd = self.to_rdd(training_datasets)
        image_paths_set = (
            # RDD(directory_path), directory containing a dataset
            training_datasets_rdd
            # RDD(file_path), paths of all label txt files
            .map(data_utils.get_all_image_paths)
            .cache())

        image_paths_set.foreach(lambda image_paths: _executor(image_paths, model_output_path))


if __name__ == "__main__":
    Yolov3Training().main()
