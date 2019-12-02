#!/usr/bin/env python3

import glob
import os

from absl import flags
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.dataset_only_image import DatasetOnlyImage
from fueling.perception.YOLOv3.inference import Inference
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client
import fueling.perception.YOLOv3.utils.data_utils as data_utils


class Yolov3Inference(BasePipeline):
    """Inference pipeline."""

    def run_test(self):
        """Run local."""
        self.run('/apollo/modules/data/fuel/testdata/perception/YOLOv3/models',
                 '/apollo/modules/data/fuel/testdata/perception/YOLOv3/models/inference_/')

    def run_prod(self):
        """Run prod."""
        object_storage = self.partner_storage() or self.our_storage()
        input_data_path = object_storage.abs_path(self.FLAGS.get('output_data_path'))
        output_data_path = os.path.join(input_data_path, cfg.inference_path)
        self.run(input_data_path, output_data_path)

    def run(self, input_dir, output_dir):
        """Run the actual pipeline job."""

        def _executor(image_paths, trained_model_path, infer_output_path):
            """Executor task that runs on workers"""
            if not image_paths:
                logging.warn('no images found in this set for inference')
                return

            logging.info(f'current image set size: {len(image_paths)}')

            restore_path = os.path.join(trained_model_path, cfg.restore_path)
            logging.info(f'infer output {infer_output_path}, restore path {restore_path}')

            engine = Inference(restore_path)
            engine.setup_network()
            if cfg.inference_only_2d:
                data_pool = DatasetOnlyImage(image_paths)
            else:
                data_pool = Dataset(image_paths)

            logging.info(f'dataset size {data_pool.dataset_size}, batch size {cfg.batch_size}')

            rounds = 0 if data_pool.dataset_size == 0 else max(
                1, (data_pool.dataset_size + 1) // cfg.batch_size)
            for _ in range(rounds):
                data = data_pool.batch
                engine.run(data, infer_output_path)

        logging.info(f'input inference data path: {input_dir}')
        logging.info(f'output inference data path: {output_dir}')

        file_utils.makedirs(os.path.join(output_dir, 'label'))
        file_utils.makedirs(os.path.join(output_dir, 'images'))
        file_utils.makedirs(os.path.join(output_dir, 'images_gt'))

        test_list_path = os.path.join(output_dir, cfg.test_list)
        image_paths_set = BasePipeline.SPARK_CONTEXT.pickleFile(test_list_path)
        image_paths_set.foreach(lambda image_paths: _executor(image_paths, input_dir, output_dir))


if __name__ == "__main__":
    Yolov3Inference().main()
