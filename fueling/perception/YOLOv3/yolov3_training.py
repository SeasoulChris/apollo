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
        data_dir = '/apollo/modules/data/fuel/testdata/perception'
        output_dir = os.path.join(data_dir, 'yolov3_output')
        training_dataset = glob.glob(os.path.join(data_dir,
                                                  'label',
                                                  '*.txt'))
        training_dataset = [(i, ele) for i, ele in enumerate(training_dataset)]
        # RDD(file_path) for training dataset.
        training_dataset_rdd = self.to_rdd(training_dataset)
        self.run(training_dataset_rdd, output_dir)

    def run_prod(self):
        dataset_dir = 'modules/perception/yolov3/training'
        # RDD(file_path) for training dataset
        training_dataset_rdd = self.to_rdd(self.bos().list_files(prefix, '.txt'))
        output_dir = bos_client.abs_path(
            'modules/perception/yolov3_output/')
        self.run(training_dataset_rdd, output_dir)

    def run(self, training_dataset_rdd, output_dir):
        data = (
            training_dataset_rdd
            .map(data_utils.get_all_paths)
            .map(data_utils.process_data)
            .map(data_utils.filter_classes)
            .cache())

        engine = training()
        engine.setup_training()

        for i in range(MAX_ITER):
            data_temp = data.lookup(i)
            image_data, y_true, cls_box_map, objs, calib = data_temp[0]
            scale1, scale2, scale3 = y_true
            image_data = np.expand_dims(image_data, axis=0)
            temp_data = (image_data, scale1, scale2, scale3, cls_box_map, objs, calib)
            engine.step(temp_data)


if __name__ == '__main__':
    Yolov3Training().main()
