#!/usr/bin/env python3

import os

import glob
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.dataset import Dataset
from fueling.perception.YOLOv3.train import training
import fueling.common.storage.bos_client as bos_client
import fueling.perception.YOLOv3.utils.data_utils as data_utils


MAX_ITER = cfg.max_iter


class Yolov3Training(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, "yolov3")

    def run_test(self):
        data_dir = "/apollo/modules/data/fuel/testdata/perception"
        training_datasets = glob.glob(os.path.join(data_dir, "*"))
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
        data_dir = "modules/perception/camera_object/"
        training_datasets = glob.glob(os.path.join("/mnt/bos", data_dir, "*"))
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
        data.foreach(_executor)


if __name__ == "__main__":
    Yolov3Training().main()
