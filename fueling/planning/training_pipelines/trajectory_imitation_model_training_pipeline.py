#!/usr/bin/env python

import os

from absl import app
from absl import flags
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
from fueling.planning.models.trajectory_imitation.conv_rnn_model import TrajectoryImitationConvRNN
from fueling.planning.models.trajectory_imitation.image_representation_loss import \
    TrajectoryImitationConvRNNLoss, \
    TrajectoryImitationCNNLSTMWithAuxiliaryEnvLoss
from fueling.planning.models.trajectory_imitation.trajectory_point_displacement_loss import \
    TrajectoryPointDisplacementMSELoss
from fueling.planning.input_feature_preprocessor.chauffeur_net_feature_generator \
    import ChauffeurNetFeatureGenerator

flags.DEFINE_string('model_type', None,
                    'model type, cnn, rnn, cnn_lstm, cnn_lstm_aux')
flags.DEFINE_string('train_set_dir', None, 'training set data folder')
flags.DEFINE_string('validation_set_dir', None, 'validation set data folder')
flags.DEFINE_string('gpu_idx', None, 'which gpu to use')
flags.DEFINE_bool('update_base_map', False,
                  'Whether to redraw the base imgs needed for training')
flags.DEFINE_list('regions_list', 'sunnyvale, san_mateo, sunnyvale_with_two_offices',
                  'maps supported for training')
flags.DEFINE_string('renderer_config_file', '/fuel/fueling/planning/input_feature_preprocessor'
                    '/planning_semantic_map_config.pb.txt',
                    'renderer configuration file in pb.txt')
flags.DEFINE_string('renderer_base_map_img_dir', '/fuel/testdata/planning/semantic_map_features',
                    'location to store map base img')
flags.DEFINE_string('renderer_base_map_data_dir', '/apollo/modules/map/data/',
                    'location to store map base img')
flags.DEFINE_bool('img_feature_rotation', True,
                  'whether to do random img rotation')
flags.DEFINE_bool('past_motion_dropout', True,
                  'whether to do past motion dropout')
flags.DEFINE_string('model_save_dir', '/fuel',
                    'specify the directory to save trained models.')


def training(model_type,
             train_set_dir,
             validation_set_dir,
             gpu_idx,
             update_base_map,
             regions_list,
             renderer_config_file,
             renderer_base_map_img_dir,
             renderer_base_map_data_dir,
             img_feature_rotation,
             past_motion_dropout,
             model_save_dir):

    # TODO(Jinyun): check performance
    cv.setNumThreads(0)

    # Set-up the GPU to use, single gpu is prefererd now because of jit issue
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

    # random number seed
    torch.manual_seed(0)

    # Set-up data-loader
    model = None
    loss = None
    train_dataset = None
    valid_dataset = None

    if update_base_map:
        ChauffeurNetFeatureGenerator.draw_base_map(regions_list,
                                                   renderer_config_file,
                                                   renderer_base_map_img_dir,
                                                   renderer_base_map_data_dir)
    renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
    renderer_config = proto_utils.get_pb_from_text_file(
        renderer_config_file, renderer_config)

    if model_type == 'cnn':
        train_dataset = TrajectoryImitationCNNFCDataset(train_set_dir,
                                                        regions_list,
                                                        renderer_config_file,
                                                        renderer_base_map_img_dir,
                                                        renderer_base_map_data_dir,
                                                        img_feature_rotation,
                                                        past_motion_dropout,
                                                        ouput_point_num=10)
        valid_dataset = TrajectoryImitationCNNFCDataset(validation_set_dir,
                                                        regions_list,
                                                        renderer_config_file,
                                                        renderer_base_map_img_dir,
                                                        renderer_base_map_data_dir,
                                                        img_feature_rotation,
                                                        past_motion_dropout,
                                                        ouput_point_num=10)
        model = TrajectoryImitationCNNFC(pred_horizon=10)
        loss = TrajectoryPointDisplacementMSELoss(4)

    elif model_type == 'rnn':
        train_dataset = TrajectoryImitationConvRNNDataset(train_set_dir,
                                                          regions_list,
                                                          renderer_config_file,
                                                          renderer_base_map_img_dir,
                                                          renderer_base_map_data_dir,
                                                          img_feature_rotation,
                                                          past_motion_dropout,
                                                          ouput_point_num=10)
        valid_dataset = TrajectoryImitationConvRNNDataset(validation_set_dir,
                                                          regions_list,
                                                          renderer_config_file,
                                                          renderer_base_map_img_dir,
                                                          renderer_base_map_data_dir,
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
        loss = TrajectoryImitationConvRNNLoss(pos_dist_loss_weight=1,
                                              box_loss_weight=1,
                                              pos_reg_loss_weight=1)
        # loss = TrajectoryImitationConvRNNWithEnvLoss(pos_dist_loss_weight=1,
        #                                              box_loss_weight=1,
        #                                              pos_reg_loss_weight=1,
        #                                              collision_loss_weight=1,
        #                                              offroad_loss_weight=1,
        #                                              onrouting_loss_weight=1,
        #                                              imitation_dropout=False)

    elif model_type == 'cnn_lstm':
        train_dataset = TrajectoryImitationCNNLSTMDataset(train_set_dir,
                                                          regions_list,
                                                          renderer_config_file,
                                                          renderer_base_map_img_dir,
                                                          renderer_base_map_data_dir,
                                                          img_feature_rotation,
                                                          past_motion_dropout,
                                                          history_point_num=10,
                                                          ouput_point_num=10)
        valid_dataset = TrajectoryImitationCNNLSTMDataset(validation_set_dir,
                                                          regions_list,
                                                          renderer_config_file,
                                                          renderer_base_map_img_dir,
                                                          renderer_base_map_data_dir,
                                                          img_feature_rotation,
                                                          past_motion_dropout,
                                                          history_point_num=10,
                                                          ouput_point_num=10)
        model = TrajectoryImitationCNNLSTM(history_len=10, pred_horizon=10, embed_size=64,
                                           hidden_size=128)
        loss = TrajectoryPointDisplacementMSELoss(4)

    elif model_type == 'cnn_lstm_aux':
        train_dataset = TrajectoryImitationCNNLSTMWithAENDataset(train_set_dir,
                                                                 regions_list,
                                                                 renderer_config_file,
                                                                 renderer_base_map_img_dir,
                                                                 renderer_base_map_data_dir,
                                                                 img_feature_rotation,
                                                                 past_motion_dropout,
                                                                 history_point_num=10,
                                                                 ouput_point_num=10)
        valid_dataset = TrajectoryImitationCNNLSTMWithAENDataset(validation_set_dir,
                                                                 regions_list,
                                                                 renderer_config_file,
                                                                 renderer_base_map_img_dir,
                                                                 renderer_base_map_data_dir,
                                                                 img_feature_rotation,
                                                                 past_motion_dropout,
                                                                 history_point_num=10,
                                                                 ouput_point_num=10)
        model = TrajectoryImitationCNNLSTMWithAuxilaryEvaluationNet(input_img_size=[
            renderer_config.height,
            renderer_config.width],
            history_len=10,
            pred_horizon=10,
            img_resolution=renderer_config.resolution,
            initial_box_x_idx=renderer_config.ego_idx_x,
            initial_box_y_idx=renderer_config.ego_idx_y,
            vehicle_front_edge_to_center=3.89,
            vehicle_back_edge_to_center=1.043,
            vehicle_width=1.055,
            embed_size=64,
            hidden_size=128)
        loss = TrajectoryImitationCNNLSTMWithAuxiliaryEnvLoss(box_loss_weight=1,
                                                              pos_reg_loss_weight=1,
                                                              collision_loss_weight=1,
                                                              offroad_loss_weight=1,
                                                              onrouting_loss_weight=1,
                                                              imitation_dropout=False)
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
        logging.warning("multiple GPUs are used, but not suggested")
        model = torch.nn.DataParallel(model)

    # Model training:
    torch.autograd.set_detect_anomaly(True)

    train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs=50, save_name=model_save_dir, print_period=50)


def main(argv):
    gflag = flags.FLAGS
    model_type = gflag.model_type
    train_set_dir = gflag.train_set_dir
    validation_set_dir = gflag.validation_set_dir
    gpu_idx = gflag.gpu_idx
    update_base_map = gflag.update_base_map
    regions_list = gflag.regions_list
    renderer_config_file = gflag.renderer_config_file
    renderer_base_map_img_dir = gflag.renderer_base_map_img_dir
    renderer_base_map_data_dir = gflag.renderer_base_map_data_dir
    img_feature_rotation = gflag.img_feature_rotation
    past_motion_dropout = gflag.past_motion_dropout
    model_save_dir = gflag.model_save_dir

    training(model_type,
             train_set_dir,
             validation_set_dir,
             gpu_idx,
             update_base_map,
             regions_list,
             renderer_config_file,
             renderer_base_map_img_dir,
             renderer_base_map_data_dir,
             img_feature_rotation,
             past_motion_dropout,
             model_save_dir)


if __name__ == "__main__":
    app.run(main)
