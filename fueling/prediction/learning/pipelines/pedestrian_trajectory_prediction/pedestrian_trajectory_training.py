#!/usr/bin/env python

import os
import sys
import time

import torch

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
from fueling.learning.train_utils import train_valid_dataloader
from fueling.prediction.learning.pipelines.pedestrian_trajectory_prediction \
    .pedestrian_trajectory_dataset_cloud import PedestrianTrajectoryDatasetCloud
from fueling.prediction.learning.models.semantic_map_model.semantic_map_model \
    import SemanticMapSelfLSTMModel, WeightedSemanticMapLoss


class PedestrianTraining(BasePipeline):
    def __init__(self, region):
        super(PedestrianTraining, self).__init__()
        self.region = region

    def run(self):
        """Run."""
        train_folder = "/mnt/bos/modules/prediction/kinglong_train_clean/" + \
                       "baidudasha/jinlong-JinLongBaiduDaSha/20200226/"
        # train_folder = "/fuel/kinglong_train_clean_split/"
        time_start = time.time()
        self.to_rdd(range(1)).foreach(lambda instance: self.train(instance, train_folder))
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))

    def train(self, instance_id, train_folder):
        """Run training task"""
        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

        # Use gpu0 for training
        # device = torch.device('cuda:0')

        dataset = PedestrianTrajectoryDatasetCloud(train_folder, self.region)
        valid_size = dataset.__len__() // 5
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset, [dataset.__len__() - valid_size, valid_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,
                                                   num_workers=1, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True,
                                                   num_workers=1, drop_last=True)
        model = SemanticMapSelfLSTMModel(30, 20).cuda()
        loss = WeightedSemanticMapLoss()
        learning_rate = 3e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.3, patience=3, min_lr=1e-9, verbose=True, mode='min')
        # Model training:
        train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer, scheduler,
                               epochs=30, save_name='/fuel/', print_period=10, save_mode=1)


if __name__ == '__main__':
    PedestrianTraining('baidusasha').main()
