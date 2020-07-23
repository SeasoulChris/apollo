#!/usr/bin/env python

import argparse
import os

import cv2 as cv
import torch

from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from fueling.learning.train_utils import train_valid_dataloader
from fueling.planning.datasets.img_in_traj_out_dataset \
    import TrajectoryImitationCNNFCDataset, \
    TrajectoryImitationConvRNNDataset, \
    TrajectoryImitationCNNLSTMDataset, \
    TrajectoryImitationCNNLSTMWithAENDataset
from fueling.planning.models.trajectory_imitation.cnn_fc_model import \
    TrajectoryImitationCNNFC
from fueling.planning.models.trajectory_imitation.cnn_lstm_model import \
    TrajectoryImitationCNNLSTM,\
    TrajectoryImitationCNNLSTMWithAuxilaryEvaluationNet
from fueling.planning.models.trajectory_imitation.conv_rnn_model import \
    TrajectoryImitationConvRNN, \
    TrajectoryImitationDeeperConvRNN, \
    TrajectoryImitationConvRNNUnetResnet18v1, \
    TrajectoryImitationConvRNNUnetResnet18v2
from fueling.planning.models.trajectory_imitation.image_representation_loss import \
    TrajectoryImitationConvRNNLoss, \
    TrajectoryImitationConvRNNWithEnvLoss, \
    TrajectoryImitationCNNLSTMWithAuxiliaryEnvLoss
from fueling.planning.models.trajectory_imitation.trajectory_point_displacemet_loss import \
    TrajectoryPointDisplacementMSELoss, \
    TrajectoryPointDisplacementL1Loss


def training(model_type, train_dir, valid_dir, renderer_config_file,
             renderer_base_map_dir, img_feature_rotation, past_motion_dropout, model_save_dir,
             region, map_path):
    logging.info(
        'training directory:{} validation directory:{}'.format(train_dir, valid_dir))

    # random number seed
    torch.manual_seed(0)

    # Set-up data-loader
    model = None
    loss = None
    train_dataset = None
    valid_dataset = None

    renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
    renderer_config = proto_utils.get_pb_from_text_file(
        renderer_config_file, renderer_config)

    if model_type == 'cnn':
        train_dataset = TrajectoryImitationCNNFCDataset(train_dir,
                                                        renderer_config_file,
                                                        renderer_base_map_dir,
                                                        map_path,
                                                        region,
                                                        img_feature_rotation,
                                                        past_motion_dropout,
                                                        ouput_point_num=10)
        valid_dataset = TrajectoryImitationCNNFCDataset(valid_dir,
                                                        renderer_config_file,
                                                        renderer_base_map_dir,
                                                        map_path,
                                                        region,
                                                        img_feature_rotation,
                                                        past_motion_dropout,
                                                        ouput_point_num=10)
        model = TrajectoryImitationCNNFC(pred_horizon=10)
        loss = TrajectoryPointDisplacementMSELoss(4)

    elif model_type == 'rnn':
        train_dataset = TrajectoryImitationConvRNNDataset(train_dir,
                                                          renderer_config_file,
                                                          renderer_base_map_dir,
                                                          map_path,
                                                          region,
                                                          img_feature_rotation,
                                                          past_motion_dropout,
                                                          ouput_point_num=10)
        valid_dataset = TrajectoryImitationConvRNNDataset(valid_dir,
                                                          renderer_config_file,
                                                          renderer_base_map_dir,
                                                          map_path,
                                                          region,
                                                          img_feature_rotation,
                                                          past_motion_dropout,
                                                          ouput_point_num=10)
        model = TrajectoryImitationConvRNN(
            input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        # model = TrajectoryImitationDeeperConvRNN(
        #     input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        # model = TrajectoryImitationConvRNNUnetResnet18v1(
        #     input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        # model = TrajectoryImitationConvRNNUnetResnet18v2(
        #     input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        loss = TrajectoryImitationConvRNNLoss(1, 1, 1)
        # loss = TrajectoryImitationConvRNNWithEnvLoss(1, 1, 1, 1, 1, True)

    elif model_type == 'cnn_lstm':
        train_dataset = TrajectoryImitationCNNLSTMDataset(train_dir,
                                                          renderer_config_file,
                                                          renderer_base_map_dir,
                                                          map_path,
                                                          region,
                                                          img_feature_rotation,
                                                          past_motion_dropout,
                                                          history_point_num=10,
                                                          ouput_point_num=10)
        valid_dataset = TrajectoryImitationCNNLSTMDataset(valid_dir,
                                                          renderer_config_file,
                                                          renderer_base_map_dir,
                                                          map_path,
                                                          region,
                                                          img_feature_rotation,
                                                          past_motion_dropout,
                                                          history_point_num=10,
                                                          ouput_point_num=10)
        model = TrajectoryImitationCNNLSTM(history_len=10, pred_horizon=10, embed_size=64,
                                           hidden_size=128)
        loss = TrajectoryPointDisplacementMSELoss(4)

    elif model_type == 'cnn_lstm_aux':
        train_dataset = TrajectoryImitationCNNLSTMWithAENDataset(train_dir,
                                                                 renderer_config_file,
                                                                 renderer_base_map_dir,
                                                                 map_path,
                                                                 region,
                                                                 img_feature_rotation,
                                                                 past_motion_dropout,
                                                                 history_point_num=10,
                                                                 ouput_point_num=10)
        valid_dataset = TrajectoryImitationCNNLSTMWithAENDataset(valid_dir,
                                                                 renderer_config_file,
                                                                 renderer_base_map_dir,
                                                                 map_path,
                                                                 region,
                                                                 img_feature_rotation,
                                                                 past_motion_dropout,
                                                                 history_point_num=10,
                                                                 ouput_point_num=10)
        model = TrajectoryImitationCNNLSTMWithAuxilaryEvaluationNet(input_img_size=[
            renderer_config.height,
            renderer_config.width],
            history_len=10,
            pred_horizon=10,
            embed_size=64,
            hidden_size=128)
        loss = TrajectoryImitationCNNLSTMWithAuxiliaryEnvLoss()
    else:
        logging.info('model {} is not implemnted'.format(model_type))
        exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,
                                               num_workers=8, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=True,
                                               num_workers=8, drop_last=True)

    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=3, min_lr=1e-9, verbose=True, mode='min')

    # if train from a trained model
    # model_file = ""
    # model.load_state_dict(torch.load(model_file))

    # CUDA setup:
    if torch.cuda.is_available():
        logging.info("Using CUDA to speed up training.")
        model.cuda()
    else:
        logging.info("Not using CUDA.")

    # Not suggested right now as jit trace can't trace a nn.DataParallel module
    if torch.cuda.device_count() > 1:
        logging.info("multiple GPUs are used")
        model = torch.nn.DataParallel(model)

    # Model training:
    torch.autograd.set_detect_anomaly(True)

    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=50, save_name=model_save_dir, print_period=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('model_type', type=str,
                        help='model type, cnn, rnn, cnn_lstm, cnn_lstm_aux')
    parser.add_argument('train_file', type=str, help='training data')
    parser.add_argument('valid_file', type=str, help='validation data')
    parser.add_argument('gpu_idx', type=str, help='which gpu to use')
    parser.add_argument('-renderer_config_file', '--renderer_config_file',
                        type=str, default='/fuel/fueling/planning/input_feature_preprocessor'
                                          '/planning_semantic_map_config.pb.txt',
                        help='renderer configuration file in proto.txt')
    parser.add_argument('-renderer_base_map_dir', '--renderer_base_map_dir', type=str,
                        default='/fuel/testdata/planning/semantic_map_features',
                        help='location to store input base img or output img')
    parser.add_argument('-img_feature_rotation', '--img_feature_rotation', type=bool,
                        default=True, help='whether to do random img rotation')
    parser.add_argument('-past_motion_dropout', '--past_motion_dropout', type=bool,
                        default=True, help='whether to do past motion dropout')
    parser.add_argument('-save_dir', '--save_dir', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    # TODO(Jinyun): check performance
    cv.setNumThreads(0)

    # Set-up the GPU to use, single gpu is prefererd now because of jit issue
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx

    region = "sunnyvale_with_two_offices"
    map_path = "/apollo/modules/map/data/" + region + "/base_map.bin"

    training(args.model_type,
             args.train_file,
             args.valid_file,
             args.renderer_config_file,
             args.renderer_base_map_dir,
             args.img_feature_rotation,
             args.past_motion_dropout,
             args.save_dir,
             region,
             map_path)
