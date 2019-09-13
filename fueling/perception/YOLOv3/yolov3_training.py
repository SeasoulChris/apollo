#!/usr/bin/env python3

import os

import glob
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.train import training
import fueling.perception.YOLOv3.utils.data_utils as data_utils
import fueling.common.storage.bos_client as bos_client


MAX_ITER = cfg.max_iter


class Yolov3Training(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'yolov3')

    def run_test(self):
        def _get_all_label_txt_paths(dataset_path):
            label_dir = os.path.join(dataset_path, "label")
            txt_list = glob.glob(os.path.join(label_dir, "*.txt"))
            return txt_list

        data_dir = '/apollo/modules/data/fuel/testdata/perception'
        training_datasets = glob.glob(os.path.join(data_dir, "*"))
        # RDD(file_path) for training dataset.
        training_datasets_rdd = self.to_rdd(training_datasets)
        data = (
            training_datasets_rdd
            .map(_get_all_label_txt_paths)
            .cache())
        output_dir = os.path.join(data_dir, 'yolov3_output')
        self.run(training_datasets_rdd, output_dir)

    def run_prod(self):
        def _get_all_label_txt_paths_bos(dataset_path):
            label_dir = os.path.join(dataset_path, "label")
            txt_list = self.bos().list_files(label_dir, ".txt", to_abs_path=False)
            txt_list = [os.path.join('/mnt/bos/', path) for path in txt_list]
            return txt_list

        data_dir = 'modules/perception/camera_object/'
        training_datasets = self.bos().list_dirs(data_dir, to_abs_path=False)
        training_datasets = [path for path in training_datasets if len(path.split("/"))==4]
        # RDD(file_path) for training dataset.
        training_datasets_rdd = self.to_rdd(training_datasets)
        data = (
            training_datasets_rdd
            .map(_get_all_label_txt_paths_bos)
            .cache())
        output_dir = os.path.join(data_dir, 'yolov3_output')
        self.run(data, output_dir)

    def run(self, data, output_dir):
        def _executor(label_txt_paths):
            engine = training()
            engine.setup_training()
            for i in range(MAX_ITER):
                txt_idx = i % len(label_txt_paths)
                label_txt_path = label_txt_paths[txt_idx]
                all_paths = data_utils.get_all_paths(label_txt_path)
                processed_data = data_utils.process_data(all_paths)
                image_data, y_true, cls_box_map, objs, calib = \
                        data_utils.filter_classes(processed_data)
                scale1, scale2, scale3 = y_true
                image_data = np.expand_dims(image_data, axis=0)
                temp_data = (image_data, scale1, scale2, scale3, cls_box_map, objs, calib)
                engine.step(temp_data)

        data.foreach(_executor)

if __name__ == '__main__':
    Yolov3Training().main()
