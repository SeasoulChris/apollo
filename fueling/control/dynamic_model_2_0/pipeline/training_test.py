#!/usr/bin/env python
import pickle
import os
import time

from torch.utils.data import DataLoader
import numpy as np
import torch

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config
from fueling.control.dynamic_model_2_0.conf.model_conf import training_config
from fueling.control.dynamic_model_2_0.conf.model_conf import smoke_test_training_config
from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset import BosSetDataset
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_state_dict
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_torch_script
import fueling.common.distributed_data_parallel as ddp
import fueling.common.logging as logging
import fueling.control.dynamic_model_2_0.gp_regression.train_utils as train_utils

SMOKE_TEST = True
if SMOKE_TEST:
    config = smoke_test_training_config
    training_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/smoke_test_2/train"
    validation_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/smoke_test_2/test"
    result_data_path = "/mnt/bos/modules/control/dynamic_model_2_0/smoke_test_2/result"
    param_file_path = os.path.join(training_data_path, "param.bin")
else:
    config = training_config
    platform_path = "/mnt/bos/modules/control/dynamic_model_2_0"
    training_data_path = os.path.join(
        platform_path, "splitted_file_list/2020-07-06-21_0806_3/train")
    validation_data_path = os.path.join(
        platform_path, "splitted_file_list/2020-07-06-21_0806_3/test")
    result_data_path = os.path.join(platform_path, "splitted_file_list/2020-07-06-21_0806_3/result")
    param_file_path = os.path.join(
        platform_path, "splitted_file_list/2020-07-06-21_0806_3/train/param.bin")
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
            'Generate validation data set in'
            + F'{time_end_validation_set - time_begin_validation_set} seconds.')

        train_files = object_storage.list_files(training_data_path, suffix='.txt')
        train_set = set()
        for cur_file in train_files:
            logging.info(cur_file)
            with open(cur_file, "rb") as fp:  # Pickling
                train_set.update(pickle.load(fp))

        # get dataset
        train_dataset = BosSetDataset(
            storage=object_storage, input_data_dir=input_data_path,
            exclude_set=validation_set, is_train=True, param_file=param_file_path)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=True, drop_last=True)
        time_end_train_dataset = time.time()
        logging.info(
            F'train loader ready in {time_end_train_dataset - time_end_validation_set} seconds.')
        total_train_number = len(train_loader.dataset)
        logging.info(total_train_number)

        # inducing points
        step_size = int(max(config["batch_size"] / config["num_inducing_point"], 1))
        inducing_point_num = torch.arange(0, config["batch_size"], step=step_size)
        for idx, (features, labels) in enumerate(train_loader):
            features = torch.transpose(features, 0, 1).type(torch.FloatTensor)
            inducing_points = features[:, inducing_point_num, :].unsqueeze(0)
            break

        inducing_points = torch.cat((inducing_points, inducing_points), 0)
        logging.info(inducing_points.shape)

        # validate loader
        valid_dataset = BosSetDataset(
            storage=object_storage, input_data_dir=input_data_path,
            exclude_set=train_dataset, is_train=False, param_file=param_file_path)
        # reduce batch size when memory is not enough len(valid_dataset.datasets)
        valid_loader = DataLoader(valid_dataset, batch_size=1024)

        encoder_net_model = Encoder(u_dim=feature_config["input_dim"],
                                    kernel_dim=config["kernel_dim"])

        logging.info(
            F'train loader ready in {time.time() - time_end_train_dataset} seconds.')
        time_prev = time.time()
        model, likelihood, optimizer, loss_fn = train_utils.init_train(
            inducing_points, encoder_net_model, feature_config["output_dim"],
            total_train_number, config["lr"], kernel_dim=config["kernel_dim"])

        logging.info(F'train init done {time.time()-time_prev}')
        time_prev = time.time()

        train_loss_plot = os.path.join(result_folder, 'train_loss.png')

        for idx, (test_features, test_labels) in enumerate(valid_loader):
            test_features = torch.transpose(test_features, 0, 1).type(torch.FloatTensor)
            break

        logging.info(F'test feature done {time.time()-time_prev}')
        time_prev = time.time()

        model, likelihood, final_train_loss = train_utils.train_save_best_model(
            config["num_epochs"], train_loader, model, likelihood,
            loss_fn, optimizer, test_features, result_folder,
            fig_file_path=train_loss_plot, is_transpose=True)

        print(f'final train loss is {final_train_loss}')
        # test save and load model
        save_model_state_dict(model, likelihood, offline_model_path)
        # save model as jit script
        save_model_torch_script(model, likelihood, test_features, online_model_path)
        # save inducing points for load model
        np.save(os.path.join(result_folder, 'inducing_points.npy'), inducing_points)


if __name__ == '__main__':
    Training().main()
