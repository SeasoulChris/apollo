#!/usr/bin/env python

import argparse
import os

import torch

from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.logging as logging
from fueling.learning.train_utils import train_valid_dataloader
from fueling.planning.datasets.img_in_traj_out_dataset \
    import TrajectoryImitationCNNDataset, TrajectoryImitationRNNDataset
from fueling.planning.models.trajectory_imitation_model \
    import TrajectoryImitationCNNModel, TrajectoryImitationCNNLoss, \
    TrajectoryImitationRNNModel, TrajectoryImitationRNNLoss, TrajectoryImitationWithEnvRNNLoss
import fueling.common.proto_utils as proto_utils


if __name__ == "__main__":
    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    # data parser:
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('model_type', type=str, help='model type, cnn or rnn')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('-renderer_config_file', '--renderer_config_file',
                        type=str, default='/fuel/fueling/planning/datasets/'
                        'semantic_map_feature/planning_semantic_map_config.pb.txt',
                        help='renderer configuration file in proto.txt')
    parser.add_argument('-imgs_dir', '--imgs_dir', type=str, default='/fuel/testdata/'
                        'planning/semantic_map_features',
                        help='location to store input base img or output img')
    parser.add_argument('-input_data_augmentation', '--input_data_augmentation', type=bool,
                        default=False, help='whether to do input data augmentation')
    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    logging.info(
        'training directory:{} validation directory:{}'.format(
            args.train_file,
            args.valid_file))

    # Set-up data-loader
    model = None
    loss = None
    train_dataset = None
    valid_dataset = None

    renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
    renderer_config = proto_utils.get_pb_from_text_file(
        args.renderer_config_file, renderer_config)

    if args.model_type == 'cnn':
        train_dataset = TrajectoryImitationCNNDataset(args.train_file,
                                                      args.renderer_config_file,
                                                      args.imgs_dir,
                                                      args.input_data_augmentation)
        valid_dataset = TrajectoryImitationCNNDataset(args.valid_file,
                                                      args.renderer_config_file,
                                                      args.imgs_dir,
                                                      args.input_data_augmentation)
        model = TrajectoryImitationCNNModel(pred_horizon=10)
        loss = TrajectoryImitationCNNLoss()

    elif args.model_type == 'rnn':
        train_dataset = TrajectoryImitationRNNDataset(args.train_file,
                                                      args.renderer_config_file,
                                                      args.imgs_dir,
                                                      args.input_data_augmentation)
        valid_dataset = TrajectoryImitationRNNDataset(args.valid_file,
                                                      args.renderer_config_file,
                                                      args.imgs_dir,
                                                      args.input_data_augmentation)
        model = TrajectoryImitationRNNModel(
            input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        loss = TrajectoryImitationRNNLoss()

    else:
        logging.info('model {} is not implemnted'.format(args.model_type))
        exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,
                                               num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True,
                                               num_workers=4, drop_last=True)

    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=3, min_lr=1e-9, verbose=True, mode='min')

    # CUDA setup:
    if torch.cuda.is_available():
        logging.info("Using CUDA to speed up training.")
        model.cuda()
    else:
        logging.info("Not using CUDA.")

    if torch.cuda.device_count() > 1:
        logging.info("multiple GPUs are used")
        model = torch.nn.DataParallel(model)

    # Model training:
    torch.autograd.set_detect_anomaly(True)

    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=10, save_name='./', print_period=50)
