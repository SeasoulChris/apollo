#!/usr/bin/env python3

import os
import shutil

import glob
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.train import training
import fueling.common.storage.bos_client as bos_client
import fueling.perception.YOLOv3.utils.data_utils as data_utils


MAX_ITER = cfg.max_iter
TRAIN_DATA_DIR_LOCAL = cfg.train_data_dir_local
TRAIN_DATA_DIR_CLOUD = cfg.train_data_dir_cloud
MODEL_OUTPUT_PATH = cfg.model_output_path


class Yolov3Training(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, "yolov3_training")

    def run_test(self):
        training_datasets = glob.glob(os.path.join(TRAIN_DATA_DIR_LOCAL, "*"))
        # RDD(file_path) for training dataset.
        training_datasets_rdd = self.to_rdd(training_datasets)
        data = (
            # RDD(directory_path), directory containing a dataset
            training_datasets_rdd
            # RDD(file_path), paths of all label txt files
            .map(data_utils.get_all_image_paths)
            .cache())
        self.run(data)

    def run_prod(self):
        training_datasets = glob.glob(os.path.join("/mnt/bos", TRAIN_DATA_DIR_CLOUD, "*"))
        # RDD(file_path) for training dataset.
        training_datasets_rdd = self.to_rdd(training_datasets)
        data = (
            # RDD(directory_path), directory containing a dataset
            training_datasets_rdd
            # RDD(file_path), paths of all label txt files
            .map(data_utils.get_all_image_paths)
            .cache())
        self.run(data)

    def run(self, data):
        def _executor(image_paths):
            engine = training()
            engine.setup_training()
            data_pool = Dataset(image_paths)
            for i in range(MAX_ITER):
                data = data_pool.batch
                engine.step(data)
        if not os.path.exists(MODEL_OUTPUT_PATH):
            os.makedirs(MODEL_OUTPUT_PATH)
        shutil.copyfile("/apollo/modules/data/fuel/fueling/perception/YOLOv3/config.py",
                os.path.join(MODEL_OUTPUT_PATH, "config.py"))
        data.foreach(_executor)


if __name__ == "__main__":
    Yolov3Training().main()
