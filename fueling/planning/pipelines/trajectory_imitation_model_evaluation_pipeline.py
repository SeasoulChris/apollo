#!/usr/bin/env python
import argparse
import math
import os
import shutil

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.file_utils as file_utils
from fueling.learning.train_utils import cuda
from fueling.planning.datasets.img_in_traj_out_dataset import \
    TrajectoryImitationCNNDataset, \
    TrajectoryImitationRNNDataset
from fueling.planning.models.trajectory_imitation_model import \
    TrajectoryImitationCNNModel, \
    TrajectoryImitationRNNModel, \
    TrajectoryImitationRNNMoreConvModel, \
    TrajectoryImitationRNNUnetResnet18Modelv1, \
    TrajectoryImitationRNNUnetResnet18Modelv2
from fueling.planning.datasets.semantic_map_feature.agent_poses_future_img_renderer import \
    AgentPosesFutureImgRenderer
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils


def calculate_cnn_displacement_error(pred, y):
    batch_size = pred.size(0)
    out = (pred - y).view(batch_size, -1, 4)
    pose_diff = out[:, :, 0:2]
    heading_diff = out[:, :, 2]
    v_diff = out[:, :, 3]

    displacement_error = torch.mean(torch.sqrt(
        torch.sum(pose_diff ** 2, dim=-1))).item()
    heading_error = torch.mean(torch.abs(heading_diff)).item()
    v_error = torch.mean(torch.abs(v_diff)).item()
    return displacement_error, heading_error, v_error


def visualize_cnn_result(renderer, img_feature, coordinate_heading,
                         message_timestamp_sec, pred_points_dir, pred, y):

    for i, pred_point in enumerate(pred):
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
        true_point = y[i]
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

        print(merged_img.shape)
        print(cv.imwrite(os.path.join(pred_points_dir, "{:.3f}.png".format(
            message_timestamp_sec[i])), merged_img))


def cnn_model_evaluator(test_loader, model, renderer_config, imgs_dir):
    with torch.no_grad():
        model.eval()

        displcement_errors = []
        heading_errors = []
        v_errors = []

        output_renderer = AgentPosesFutureImgRenderer(renderer_config)

        output_dir = os.path.join(imgs_dir, "cnn_model_evaluation/")
        if os.path.isdir(output_dir):
            print(output_dir + " directory exists, delete it!")
            shutil.rmtree(output_dir)
        file_utils.makedirs(output_dir)
        print("Making output directory: " + output_dir)
        pred_points_dir = os.path.join(output_dir, 'pred_points/')
        file_utils.makedirs(pred_points_dir)

        for X, y, img_feature, coordinate_heading, message_timestamp_sec in tqdm(test_loader):
            X, y = cuda(X), cuda(y)
            pred = model(X)
            displacement_error, heading_error, v_error = \
                calculate_cnn_displacement_error(pred, y)
            displcement_errors.append(displacement_error)
            heading_errors.append(heading_error)
            v_errors.append(v_error)
            visualize_cnn_result(output_renderer, img_feature, coordinate_heading,
                                 message_timestamp_sec, pred_points_dir, pred, y)

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
                         message_timestamp_sec, pred_points_dir, explicit_memory_dir,
                         pos_dists_dir, pos_boxs_dir, pred, y):

    batched_pred_points = pred[2]
    for i, pred_point in enumerate(batched_pred_points):
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

        cv.imwrite(os.path.join(pred_points_dir, "{:.3f}.png".format(
            message_timestamp_sec[i])), merged_img)

    batched_explicit_memory = pred[3]
    for i, M_B in enumerate(batched_explicit_memory):
        position_memory_mat = M_B[0].cpu().numpy() * 255
        box_memory_mat = M_B[1].cpu().numpy() * 255
        visual_mat = np.concatenate(
            (position_memory_mat, box_memory_mat), axis=1)
        cv.imwrite(os.path.join(explicit_memory_dir,
                                "final_explicit_memory_@_{:.3f}.png".
                                format(message_timestamp_sec[i])),
                   visual_mat)

    batched_pred_pos_dists = pred[0]
    for i, pred_pos_dist in enumerate(batched_pred_pos_dists):
        origianl_shape = pred_pos_dist[0].shape
        last_mat = np.zeros((origianl_shape[1],
                             0,
                             origianl_shape[0]))
        for t, single_frame_pos in enumerate(pred_pos_dist):
            # original img pixel range is from 0 to 1,
            # multiply by 255 to better visualize it
            cur_mat = single_frame_pos.view(origianl_shape[1],
                                            origianl_shape[2],
                                            origianl_shape[0]).cpu().numpy() * 255
            last_mat = np.concatenate((last_mat, cur_mat), axis=1)
        cv.imwrite(os.path.join(pos_dists_dir,
                                "pred_pos_dists_@_{:.3f}.png".
                                format(message_timestamp_sec[i])),
                   last_mat)

    batched_pred_boxs = pred[1]
    for i, pred_box in enumerate(batched_pred_boxs):
        origianl_shape = pred_box[0].shape
        last_mat = np.zeros((origianl_shape[1],
                             0,
                             origianl_shape[0]))
        for t, single_frame_box in enumerate(pred_box):
            # original img pixel range is from 0 to 1,
            # multiply by 255 to better visualize it
            cur_mat = single_frame_box.view(origianl_shape[1],
                                            origianl_shape[2],
                                            origianl_shape[0]).cpu().numpy() * 255
            last_mat = np.concatenate((last_mat, cur_mat), axis=1)
        cv.imwrite(os.path.join(pos_boxs_dir,
                                "pred_boxs_@_{:.3f}.png".format(message_timestamp_sec[i])),
                   last_mat)


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
        file_utils.makedirs(output_dir)
        print("Making output directory: " + output_dir)
        pred_points_dir = os.path.join(output_dir, 'pred_points/')
        file_utils.makedirs(pred_points_dir)
        explicit_memory_dir = os.path.join(output_dir, 'explicit_memory/')
        file_utils.makedirs(explicit_memory_dir)
        pos_dists_dir = os.path.join(output_dir, 'pred_pos_dists/')
        file_utils.makedirs(pos_dists_dir)
        pos_boxs_dir = os.path.join(output_dir, 'pred_boxs/')
        file_utils.makedirs(pos_boxs_dir)

        for X, y, img_feature, coordinate_heading, message_timestamp_sec in tqdm(test_loader):
            X, y = cuda(X), cuda(y)
            pred = model(X)
            displacement_error, heading_error, v_error = \
                calculate_rnn_displacement_error(pred, y)
            displcement_errors.append(displacement_error)
            heading_errors.append(heading_error)
            v_errors.append(v_error)
            visualize_rnn_result(
                output_renderer, img_feature, coordinate_heading,
                message_timestamp_sec, pred_points_dir, explicit_memory_dir,
                pos_dists_dir, pos_boxs_dir, pred, y)

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


def evaluating(model_type, model_file, test_set_folder, renderer_config_file,
               imgs_dir, region, map_path):
    model = None
    test_dataset = None

    renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
    renderer_config = proto_utils.get_pb_from_text_file(
        renderer_config_file, renderer_config)

    if model_type == 'cnn':
        model = TrajectoryImitationCNNModel(pred_horizon=10)
        test_dataset = TrajectoryImitationCNNDataset(test_set_folder,
                                                     renderer_config_file,
                                                     imgs_dir,
                                                     map_path,
                                                     region,
                                                     input_data_agumentation=False,
                                                     evaluate_mode=True)
    elif model_type == 'rnn':
        model = TrajectoryImitationRNNModel(
            input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        test_dataset = TrajectoryImitationRNNDataset(test_set_folder,
                                                     renderer_config_file,
                                                     imgs_dir,
                                                     map_path,
                                                     region,
                                                     input_data_agumentation=False,
                                                     evaluate_mode=True)
    else:
        logging.info('model {} is not implemnted'.format(model_type))
        exit()

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=2,
                                              drop_last=True)
    model_state_dict = torch.load(model_file)

    # added if model was trained using nn.DataParallel
    if args.multi_gpu_trained:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(model_state_dict)

    # CUDA setup:
    if torch.cuda.is_available():
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    if model_type == 'cnn':
        cnn_model_evaluator(test_loader, model,
                            renderer_config_file, imgs_dir)
    elif model_type == 'rnn':
        rnn_model_evaluator(test_loader, model,
                            renderer_config_file, imgs_dir)
    else:
        logging.info('model {} is not implemnted'.format(model_type))
        exit()


if __name__ == "__main__":
    # TODO(Jinyun): check performance
    cv.setNumThreads(0)

    # Set-up the GPU to use, single gpu is prefererd now because of jit issue
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    parser.add_argument('-multi_gpu_trained', '--multi_gpu_trained', type=bool,
                        default=False, help='whether trained with multi-gpu')
    parser.add_argument('-save_path', '--save_path', type=str, default='./',
                        help='Specify the directory to save trained models.')
    args = parser.parse_args()

    region = "sunnyvale_with_two_offices"
    map_path = "/apollo/modules/map/data/" + region + "/base_map.bin"

    evaluating(args.model_type, args.model_file, args.test_set_folder, args.renderer_config_file,
               args.imgs_dir, region, map_path)
