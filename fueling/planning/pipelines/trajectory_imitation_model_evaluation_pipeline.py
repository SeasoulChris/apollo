#!/usr/bin/env python
import argparse
import math
import os
import shutil

import cv2 as cv
import numpy as np
import torch

from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from fueling.learning.train_utils import cuda
from fueling.planning.datasets.img_in_traj_out_dataset import TrajectoryImitationCNNDataset, TrajectoryImitationRNNDataset
from fueling.planning.models.trajectory_imitation_model import TrajectoryImitationCNNModel, TrajectoryImitationRNNModel
from fueling.planning.datasets.semantic_map_feature.agent_poses_future_img_renderer import AgentPosesFutureImgRenderer
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils


def calculate_cnn_displacement_error(pred, y):
    y_true = y.view(y.size(0), -1)
    out = pred - y_true
    # with 5 properties x,y,phi,v, a
    out.view(-1, 5)
    pos_x_diff = out[:, 0]
    pos_y_diff = out[:, 1]
    phi_diff = out[:, 2]
    v_diff = out[:, 3]
    a_diff = out[:, 4]
    displacement_error = torch.mean(
        torch.sqrt(pos_x_diff ** 2 + pos_y_diff ** 2)).item()
    heading_error = torch.mean(phi_diff).item()
    v_error = torch.mean(torch.abs(v_diff)).item()
    a_error = torch.mean(torch.abs(a_diff)).item()
    return displacement_error, heading_error, v_error, a_error


def cnn_model_evaluator(test_loader, model):
    with torch.no_grad():
        model.eval()

        displcement_errors = []
        heading_errors = []
        v_errors = []
        a_errors = []
        for i, (X, y) in enumerate(test_loader):
            X, y = cuda(X), cuda(y)
            pred = model(X)
            displacement_error, heading_error, v_error, a_error = \
                calculate_cnn_displacement_error(pred, y)
            displcement_errors.append(displacement_error)
            heading_errors.append(heading_error)
            v_errors.append(v_error)
            a_errors.append(a_error)

        average_displacement_error = np.mean(displcement_errors)
        average_heading_error = np.mean(heading_errors)
        average_v_error = np.mean(v_errors)
        average_a_error = np.mean(a_errors)
        print('average displacement error: {}.'.format(
            average_displacement_error))
        print('average heading error: {}.'.format(average_heading_error))
        print('average speed error: {}.'.format(average_v_error))
        print('average acceleration error: {}.'.format(average_a_error))


def calculate_rnn_displacement_error(pred, y):
    pred_points = pred[2]
    true_points = y[2]
    points_diff = pred_points - true_points
    pose_diff = points_diff[:, :, 0:2]
    heading_diff = points_diff[:, :, 2]
    v_diff = points_diff[:, :, 3]

    displacement_error = torch.mean(torch.sqrt(
        torch.sum(pose_diff ** 2, dim=-1))).item()
    heading_error = torch.mean(torch.abs(heading_diff)).item()
    v_error = torch.mean(torch.abs(v_diff)).item()
    return displacement_error, heading_error, v_error


def visualize_rnn_result(renderer, img_feature, coordinate_heading,
                         message_timestamp_sec, output_dir, pred, y):

    batched_pred_point = pred[2]
    for i, pred_point in enumerate(batched_pred_point):
        # Draw pred_box in blue
        pred_box_img = renderer.draw_agent_box_future_trajectory(
            pred_point.cpu().numpy(), coordinate_heading[i], solid_box=False)
        pred_box_img = np.repeat(pred_box_img, 3, axis=2)
        pred_box_img = renderer_utils.img_white_gradient_to_color_gradient(
            pred_box_img, (255, 0, 0))

        # Draw pred_pose in pink
        pred_pose_img = renderer.draw_agent_pose_future_trajectory(
            pred_point.cpu().numpy(), coordinate_heading[i])
        pred_pose_img = np.repeat(pred_pose_img, 3, axis=2)
        pred_pose_img = renderer_utils.img_white_gradient_to_color_gradient(
            pred_pose_img, (255, 0, 255))

        # Draw true_pose in yellow
        true_point = y[2][i]
        true_pose_img = renderer.draw_agent_pose_future_trajectory(
            true_point.cpu().numpy(), coordinate_heading[i])
        true_pose_img = np.repeat(true_pose_img, 3, axis=2)
        true_pose_img = renderer_utils.img_white_gradient_to_color_gradient(
            true_pose_img, (0, 255, 255))

        merged_img = renderer_utils.img_notblack_stacking(
            pred_box_img, img_feature[i])
        merged_img = renderer_utils.img_notblack_stacking(
            true_pose_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            pred_pose_img, merged_img)

        cv.imwrite(os.path.join(output_dir, "{:.3f}".format(
            message_timestamp_sec[i]) + ".png"), merged_img)


def rnn_model_evaluator(test_loader, model, renderer_config, imgs_dir):
    with torch.no_grad():
        model.eval()

        displcement_errors = []
        heading_errors = []
        v_errors = []

        output_renderer = AgentPosesFutureImgRenderer(renderer_config)

        output_dir = os.path.join(imgs_dir, "rnn_model_evaluation/")
        if os.path.isdir(output_dir):
            print(output_dir + " directory exists, delete it!")
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        print("Making output directory: " + output_dir)

        for X, y, img_feature, coordinate_heading, message_timestamp_sec in test_loader:
            X, y = cuda(X), cuda(y)
            pred = model(X)
            displacement_error, heading_error, v_error = \
                calculate_rnn_displacement_error(pred, y)
            displcement_errors.append(displacement_error)
            heading_errors.append(heading_error)
            v_errors.append(v_error)
            visualize_rnn_result(
                output_renderer, img_feature, coordinate_heading,
                message_timestamp_sec, output_dir, pred, y)

        average_displacement_error = 'average displacement error: {}.'.format(
            np.mean(displcement_errors))
        average_heading_error = 'average heading error: {}.'.format(
            np.mean(heading_errors))
        average_v_error = 'average speed error: {}.'.format(np.mean(v_errors))
        with open(os.path.join(output_dir, "statistics.txt"), "w") as output_file:
            output_file.write(average_displacement_error + "\n")
            output_file.write(average_heading_error + "\n")
            output_file.write(average_v_error + "\n")

        print(average_displacement_error)
        print(average_heading_error)
        print(average_v_error)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('model_type', type=str, help='model type, cnn or rnn')
    parser.add_argument('model_file', type=str, help='trained model')
    parser.add_argument('test_set_folder', type=str, help='test data')
    parser.add_argument('-renderer_config_file', '--renderer_config_file',
                        type=str, default='/fuel/fueling/planning/datasets/'
                        'semantic_map_feature/planning_semantic_map_config.pb.txt',
                        help='renderer configuration file in proto.txt')
    parser.add_argument('-imgs_dir', '--imgs_dir', type=str, default='/fuel/testdata/'
                        'planning/semantic_map_features',
                        help='location to store input base img or output img')
    parser.add_argument('-input_data_augmentation', '--input_data_augmentation', type=bool,
                        default=False, help='whether to do input data augmentation')
    args = parser.parse_args()

    # Set-up the GPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = None
    test_dataset = None

    renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
    renderer_config = proto_utils.get_pb_from_text_file(
        args.renderer_config_file, renderer_config)

    if args.model_type == 'cnn':
        model = TrajectoryImitationCNNModel()
        test_dataset = TrajectoryImitationCNNDataset(args.test_set_folder,
                                                     args.renderer_config_file,
                                                     args.imgs_dir,
                                                     args.input_data_augmentation)
    elif args.model_type == 'rnn':
        model = TrajectoryImitationRNNModel(
            input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        test_dataset = TrajectoryImitationRNNDataset(args.test_set_folder,
                                                     args.renderer_config_file,
                                                     args.imgs_dir,
                                                     args.input_data_augmentation,
                                                     evaluate_mode=True)
    else:
        logging.info('model {} is not implemnted'.format(args.model_type))
        exit()

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=4,
                                              drop_last=True)
    model_state_dict = torch.load(args.model_file)

    # added because model was trained using nn.DataParallel
    model = torch.nn.DataParallel(model)

    model.load_state_dict(model_state_dict)

    # CUDA setup:
    if torch.cuda.is_available():
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    if args.model_type == 'cnn':
        cnn_model_evaluator(test_loader, model)
    elif args.model_type == 'rnn':
        rnn_model_evaluator(test_loader, model,
                            args.renderer_config_file, args.imgs_dir)
    else:
        logging.info('model {} is not implemnted'.format(args.model_type))
        exit()
