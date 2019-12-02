#!/usr/bin/env python3

import os
import shutil

from absl import flags
import glob
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.train import training
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client
import fueling.perception.YOLOv3.utils.data_utils as data_utils


class Yolov3Training(BasePipeline):
    """Model training pipeline."""

    def run_test(self):
        """Run test."""
        training_datasets = glob.glob(
            os.path.join('/apollo/modules/data/fuel/testdata/perception/YOLOv3/train', '*'))
        self.run(training_datasets, '/apollo/modules/data/fuel/testdata/perception/YOLOv3/models')

    def run_prod(self):
        """Run prod."""
        input_data_path = self.FLAGS.get('input_data_path')
        output_model_path = self.FLAGS.get('output_data_path')
        object_storage = self.partner_storage() or BosClient()
        self.run([object_storage.abs_path(input_data_path)],
                 object_storage.abs_path(output_model_path))

    def run(self, datasets, output_dir):
        """Run the actual pipeline job."""

        def _executor(image_paths, output_trained_model_path):
            """Executor task that runs on workers"""
            if not image_paths:
                logging.warn('no images found in this set')
                return

            logging.info(f'current image set size: {len(image_paths)}')

            restore_path = os.path.join(output_trained_model_path, cfg.restore_path)
            file_utils.makedirs(restore_path)
            engine = training(restore_path)
            engine.setup_training()
            data_pool = Dataset(image_paths)

            for _ in range(cfg.max_iter):
                data_batch = data_pool.batch
                engine.step(data_batch)

            # After training is done, copy the single model file over to user specified folder
            latest_model_file = data_utils.get_latest_model(restore_path, cfg.model_name_prefix)
            if latest_model_file:
                shutil.copyfile(os.path.join(restore_path, latest_model_file),
                                os.path.join(output_trained_model_path, latest_model_file))
            logging.info(
                f'moved {latest_model_file} from {restore_path} to {output_trained_model_path}')

        file_utils.makedirs(output_dir)
        logging.info(f'input training data path: {datasets}')
        logging.info(f'output trained model path: {output_dir}')

        infer_outpath = os.path.join(output_dir, cfg.inference_path)
        test_list_path = os.path.join(infer_outpath, cfg.test_list)
        if (os.path.exists(test_list_path)):
            shutil.rmtree(test_list_path)

        image_paths_set = (
            # RDD(directory_path), directory containing a dataset
            self.to_rdd(datasets)
            # RDD(file_path), paths of all label txt files
            .map(lambda data: data_utils.get_all_image_paths(data, sample=1))
            .cache())

        train_rdd = image_paths_set.map(lambda xy: xy[0])
        test_rdd = image_paths_set.map(lambda xy: xy[1])
        test_rdd.saveAsPickleFile(test_list_path)
        train_rdd.foreach(lambda image_paths: _executor(image_paths, output_dir))


if __name__ == "__main__":
    Yolov3Training().main()
