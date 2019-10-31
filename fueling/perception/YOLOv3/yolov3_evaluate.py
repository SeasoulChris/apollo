#!/usr/bin/env python3

import os

import glob
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.common.file_utils import makedirs
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.dataset_only_image import DatasetOnlyImage
from fueling.perception.YOLOv3.inference import Inference
import fueling.common.logging as logging
import fueling.common.storage.bos_client as bos_client
import fueling.perception.YOLOv3.utils.data_utils as data_utils


BATCH_SIZE = cfg.batch_size
MAX_ITER = cfg.max_iter
INFERENCE_OUTPUT_PATH = cfg.inference_output_path
INFERENCE_ONLY_2D = cfg.inference_only_2d
INFERENCE_DATA_DIR_LOCAL = cfg.inference_data_dir_local
INFERENCE_DATA_DIR_CLOUD = cfg.inference_data_dir_cloud


class Yolov3Inference(BasePipeline):

    def run_test(self):
        training_datasets = glob.glob(os.path.join(INFERENCE_DATA_DIR_LOCAL, "*"))
        # RDD(file_path) for training dataset.
        training_datasets_rdd = self.to_rdd(training_datasets)
        data = (
            # RDD(directory_path), directory containing a dataset
            training_datasets_rdd
            # RDD(file_path), paths of all label txt files
            .map(data_utils.get_all_image_paths)
            .cache())
        output_dir = os.path.join(INFERENCE_OUTPUT_PATH)
        self.run(data, output_dir)

    def run_prod(self):
        training_datasets = glob.glob(os.path.join("/mnt/bos", INFERENCE_DATA_DIR_CLOUD, "*"))
        # RDD(file_path) for training dataset.
        training_datasets_rdd = self.to_rdd(training_datasets)
        data = (
            # RDD(directory_path), directory containing a dataset
            training_datasets_rdd
            # RDD(file_path), paths of all label txt files
            .map(data_utils.get_all_image_paths)
            .cache())
        output_dir = os.path.join(INFERENCE_OUTPUT_PATH)
        self.run(data, output_dir)

    def run(self, data_rdd, output_dir):
        def _executor(image_paths):
            engine = Inference()
            engine.setup_network()
            if INFERENCE_ONLY_2D:
                data_pool = DatasetOnlyImage(image_paths)
            else:
                data_pool = Dataset(image_paths)
            for _ in range((data_pool.dataset_size + 1) // BATCH_SIZE):
                data = data_pool.batch
                engine.run(data, output_dir)
        label_output_path = os.path.join(output_dir, "label")
        images_output_path = os.path.join(output_dir, "images")
        makedirs(label_output_path)
        makedirs(images_output_path)
        data_rdd.foreach(_executor)


if __name__ == "__main__":
    Yolov3Inference().main()
