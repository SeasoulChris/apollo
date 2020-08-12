#!/usr/bin/env python
import pickle
import os
import time

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import gpytorch
import numpy as np
import torch

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config
from fueling.control.dynamic_model_2_0.conf.model_conf import smoke_test_training_config
from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset import BosSetDataset
from fueling.control.dynamic_model_2_0.gp_regression.encoder import TransformerEncoderCNN
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_state_dict
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_torch_script
import fueling.common.distributed_data_parallel as ddp
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.control.dynamic_model_2_0.gp_regression.train_utils as train_utils

config = smoke_test_training_config
# training_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/smoke_test_2/train"
# validation_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/smoke_test_2/test"
# result_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/smoke_test_2/result"

training_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/splitted_file_list/2020-07-06-21_0806_3/train"
validation_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/splitted_file_list/2020-07-06-21_0806_3/test"
result_data_path = "/mnt/bos/modules/control/splitted_file_list/2020-07-06-21_0806_set/result"
param_file_path = "/mnt/bos/modules/control/dynamic_model_2_0/splitted_file_list/2020-07-06-21_0806_3/train/param.bin"
# time
timestr = time.strftime('%Y%m%d-%H%M')
# save files at
result_folder = os.path.join(result_data_path, f'{timestr}')
offline_model_path = os.path.join(result_folder, 'gp_model.pth')
online_model_path = os.path.join(result_folder, 'gp_model.pt')
train_model = True


class Training(BasePipeline):
    def run(self):
        """Run."""
        input_data_path = self.FLAGS.get('input_data_path')
        object_storage = self.partner_storage() or self.our_storage()

        time_start = time.time()

        workers = int(os.environ.get('APOLLO_EXECUTORS', 1))
        job_id = self.FLAGS.get('job_id')
        ddp.register_job(job_id, workers)

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Spark distributing as normal
        self.to_rdd(range(workers)).foreach(lambda instance: self.train(
            instance, workers, job_id, object_storage, input_data_path))
        logging.info(F'Training complete in {time.time() - time_start} seconds.')

    @staticmethod
    def train(instance, workers, job_id, object_storage, input_data_path):
        """Run training task"""
        logging.info(F'instance: {instance}, workers: {workers}, job_id: {job_id}')

        # (self.to_rdd([abs_source_dir])
        time_begin_validation_set = time.time()

        # get validation data file set
        files = object_storage.list_files(validation_data_path, suffix='.txt')
        validation_set = set()
        for cur_file in files:
            logging.info(cur_file)
            with open(cur_file, "rb") as fp:  # Pickling
                validation_set.update(pickle.load(fp))
        time_end_validation_set = time.time()
        logging.info(
            F'Generate validation data set in {time_end_validation_set - time_begin_validation_set} seconds.')

        # get dataset
        train_dataset = BosSetDataset(storage=object_storage, input_data_dir=input_data_path,
                                      validation_set=validation_set, is_train=True, param_file=param_file_path)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, drop_last=True)
        time_end_train_dataset = time.time()
        logging.info(
            F'train loader ready in {time_end_train_dataset - time_end_validation_set} seconds.')
        total_train_number = len(train_loader.dataset)
        logging.info(total_train_number)


if __name__ == '__main__':
    Training().main()
