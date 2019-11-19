#!/usr/bin/env python3

import os
import glob
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.dataset_only_image import DatasetOnlyImage
from fueling.perception.YOLOv3.inference import Inference
from fueling.perception.YOLOv3.evaluate import Yolov3Evaluate
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
        datasets = glob.glob(os.path.join('/apollo/modules/data/fuel/testdata/perception/YOLOv3/train', '*'))
        self.run(datasets, '/apollo/modules/data/fuel/testdata/perception/YOLOv3/infer_output/models-51000/')

    def run_prod(self):
        """Run prod."""
        object_storage = self.partner_object_storage() or BosClient()
        self.run([object_storage.abs_path(self.FLAGS.get('input_inference_data_path'))],
                 object_storage.abs_path(os.path.join(self.FLAGS.get('output_inference_path'),
                                                      cfg.inference_path)))

    def run(self, datasets, output_dir):
        """Run the actual pipeline job."""

        def _executor(image_paths, output_inference_path):
            """Executor task that runs on workers"""
            if not image_paths:
                logging.warn('no images found in this set')
                return

            logging.info('current image set size: {}'.format(len(image_paths)))

            restore_path = os.path.join(output_inference_path, cfg.restore_path)
            engine = Inference(output_inference_path, restore_path)
            engine.setup_network()
            if cfg.inference_only_2d:
                data_pool = DatasetOnlyImage(image_paths)
            else:
                data_pool = Dataset(image_paths)
            for _ in range((data_pool.dataset_size + 1) // cfg.batch_size):
                data = data_pool.batch
                engine.run(data, output_dir)

        logging.info('input inference data path: {}'.format(datasets))
        logging.info('output inference data path: {}'.format(output_dir))

        # RDD(file_path) for training dataset.
        image_paths_set = (
            # RDD(directory_path), directory containing a dataset
            self.to_rdd(datasets) 
            # RDD(file_path), paths of all label txt files
            .map(data_utils.get_all_image_paths)
            .cache())
 
        file_utils.makedirs(os.path.join(output_dir, "label"))
        file_utils.makedirs(os.path.join(output_dir, "images"))
        file_utils.makedirs(os.path.join(output_dir, "images_gt"))
  
        image_paths_set.foreach(lambda image_paths: _executor(image_paths, output_dir))


if __name__ == "__main__":
    Yolov3Inference().main()
