#!/usr/bin/env python3

import os
import glob
import numpy as np
from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.dataset_only_image import DatasetOnlyImage
from fueling.perception.YOLOv3.inference import Inference
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client
import fueling.perception.YOLOv3.utils.data_utils as data_utils


flags.DEFINE_string('input_inference_data_path', '', 'Input data path for inference.')
flags.DEFINE_string('output_inference_path', '', 'Output path for inference.')


class Yolov3Inference(BasePipeline):
    """Inference pipeline."""

    def run_test(self):
        """Run local."""
        self.run('/apollo/modules/data/fuel/testdata/perception/YOLOv3/models',
                 '/apollo/modules/data/fuel/testdata/perception/YOLOv3/infer_output/models-51000/')

    def run_prod(self):
        """Run prod."""
        object_storage = self.partner_object_storage() or BosClient()
        self.run([object_storage.abs_path(self.FLAGS.get('input_inference_data_path'))],
                 object_storage.abs_path(os.path.join(self.FLAGS.get('output_inference_path'),
                                                      cfg.inference_path)))

    def run(self, input_dir, output_dir):
        """Run the actual pipeline job."""

        def _executor(image_paths, input_path, output_path):
            """Executor task that runs on workers"""
            if not image_paths:
                logging.warn('no images found in this set')
                return

            logging.info('current image set size: {}'.format(len(image_paths)))

            restore_path = os.path.join(input_path, cfg.restore_path)
            logging.info('restore path is {}'.format(restore_path))
            engine = Inference(input_path, restore_path)
            engine.setup_network()
            if cfg.inference_only_2d:
                data_pool = DatasetOnlyImage(image_paths)
            else:
                data_pool = Dataset(image_paths)
            for _ in range((data_pool.dataset_size + 1) // cfg.batch_size):
                data = data_pool.batch
                engine.run(data, output_path)

        logging.info('input inference data path: {}'.format(input_dir))
        logging.info('output inference data path: {}'.format(output_dir))
        infer_path = os.path.join(input_dir, cfg.inference_path)
        test_list_path = os.path.join(infer_path, cfg.test_list)
        image_paths_set = BasePipeline.SPARK_CONTEXT.pickleFile(test_list_path)
        file_utils.makedirs(os.path.join(output_dir, "label"))
        file_utils.makedirs(os.path.join(output_dir, "images"))
        file_utils.makedirs(os.path.join(output_dir, "images_gt"))
        image_paths_set.foreach(lambda image_paths: _executor(image_paths, input_dir, output_dir))


if __name__ == "__main__":
    Yolov3Inference().main()
