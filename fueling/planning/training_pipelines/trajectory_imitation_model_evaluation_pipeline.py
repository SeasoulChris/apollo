#!/usr/bin/env python
import os
import shutil
import math

from absl import app
from absl import flags
import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from modules.planning.proto import planning_semantic_map_config_pb2
from modules.planning.proto import planning_pb2

import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.file_utils as file_utils
from fueling.learning.train_utils import cuda
from fueling.planning.datasets.img_in_traj_out_dataset \
    import TrajectoryImitationCNNFCDataset, \
    TrajectoryImitationConvRNNDataset, \
    TrajectoryImitationSelfCNNLSTMDataset, \
    TrajectoryImitationCNNLSTMDataset
from fueling.planning.models.trajectory_imitation.cnn_fc_model import \
    TrajectoryImitationCNNFC
from fueling.planning.models.trajectory_imitation.cnn_lstm_model import \
    TrajectoryImitationSelfCNNLSTM, \
    TrajectoryImitationKinematicConstrainedCNNLSTM, \
    TrajectoryImitationUnconstrainedCNNLSTM
from fueling.planning.models.trajectory_imitation.conv_rnn_model import \
    TrajectoryImitationConvRNN
from fueling.planning.input_feature_preprocessor.agent_poses_future_img_renderer import \
    AgentPosesFutureImgRenderer
import fueling.planning.input_feature_preprocessor.renderer_utils as renderer_utils
from fueling.planning.input_feature_preprocessor.chauffeur_net_feature_generator \
    import ChauffeurNetFeatureGenerator
from fueling.planning.math_utils.math_utils import NormalizeAngle

flags.DEFINE_string('model_type', None,
                    'model type, cnn, conv_rnn, self_cnn_lstm, self_cnn_lstm_aux')
flags.DEFINE_string('model_file', None, 'trained model')
flags.DEFINE_string('test_set_dir', None, 'test set data folder')
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
flags.DEFINE_string('evaluation_output_dir', '/fuel/evaluation_output_dir',
                    'location to evaluation result')


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


WHEEL_BASE = 2.8448
MAX_STEERING = 0.6
MAX_CURVATURE = math.tan(MAX_STEERING) / WHEEL_BASE
DELTA_T = 0.2
MAX_ACCELERATION = 4
MAX_DECELERATION = -6


def calculate_kinematic_feasibility(pred, current_x, current_y, current_theta, current_v):
    batch_size = pred.shape[0]
    trajectory_horizon = pred.shape[1]

    # restore the trajectory in worldframe
    trajectory_batch = []
    # action in shape of [time_horizon, [dx, dy, dtheta, v]]
    for batch_num in range(batch_size):
        planning_message = planning_pb2.ADCTrajectory()
        current_point = planning_message.trajectory_point.add()
        current_point.path_point.x = current_x[batch_num]
        current_point.path_point.y = current_y[batch_num]
        current_point.path_point.theta = current_theta[batch_num]
        current_point.v = current_v[batch_num]

        for i in range(trajectory_horizon):
            point = planning_message.trajectory_point.add()
            dx = pred[batch_num][i][0]
            dy = pred[batch_num][i][1]
            dtheta = pred[batch_num][i][2]
            point.path_point.x = dx * math.cos(current_theta[batch_num]) - \
                dy * math.sin(current_theta[batch_num]) + current_x[batch_num]
            point.path_point.y = dx * math.sin(current_theta[batch_num]) + \
                dy * math.cos(current_theta[batch_num]) + current_y[batch_num]
            point.path_point.theta = NormalizeAngle(
                current_theta[batch_num] + dtheta)
            point.v = pred[batch_num][i][3]

        accumulated_s = 0.0
        for i in range(len(planning_message.trajectory_point) - 1):
            point = planning_message.trajectory_point[i].path_point
            nextpoint = planning_message.trajectory_point[i + 1].path_point
            point.s = accumulated_s
            accumulated_s += math.sqrt((nextpoint.x - point.x)
                                       ** 2 + (nextpoint.y - point.y) ** 2)
        for i in range(len(planning_message.trajectory_point) - 1):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            point.a = (nextpoint.v - point.v) / DELTA_T
        for i in range(len(planning_message.trajectory_point) - 1):
            point = planning_message.trajectory_point[i]
            nextpoint = planning_message.trajectory_point[i + 1]
            point.da = (nextpoint.a - point.a) / DELTA_T

        kepsilon = 1e-6
        for i in range(len(planning_message.trajectory_point) - 1):
            point = planning_message.trajectory_point[i].path_point
            nextpoint = planning_message.trajectory_point[i + 1].path_point
            point.kappa = NormalizeAngle(nextpoint.theta - point.theta) / \
                (nextpoint.s - point.s + kepsilon)
        for i in range(len(planning_message.trajectory_point) - 1):
            point = planning_message.trajectory_point[i].path_point
            nextpoint = planning_message.trajectory_point[i + 1].path_point
            point.dkappa = (nextpoint.kappa - point.kappa) / \
                (nextpoint.s - point.s + kepsilon)
        for i in range(len(planning_message.trajectory_point) - 1):
            point = planning_message.trajectory_point[i].path_point
            nextpoint = planning_message.trajectory_point[i + 1].path_point
            point.ddkappa = (nextpoint.dkappa - point.dkappa) / \
                (nextpoint.s - point.s + kepsilon)

        trajectory_batch.append(planning_message)

    # check feasibility by path curvature and max acceleration
    infeasible_num = 0
    for batch_num in range(batch_size):
        planning_message = trajectory_batch[batch_num]
        for i in range(trajectory_horizon):
            if math.fabs(planning_message.trajectory_point[i].path_point.kappa) > \
                MAX_CURVATURE or planning_message.trajectory_point[i].a > MAX_ACCELERATION or \
                    planning_message.trajectory_point[i].a < MAX_DECELERATION:
                infeasible_num += 1
                break
    return infeasible_num


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

        cv.imwrite(os.path.join(pred_points_dir, "{:.3f}.png".format(
            message_timestamp_sec[i])), merged_img)


def cnn_model_evaluator(test_loader, model, renderer_config, evaluation_output_dir):
    with torch.no_grad():
        model.eval()

        displcement_errors = []
        heading_errors = []
        v_errors = []

        output_renderer = AgentPosesFutureImgRenderer(renderer_config)

        output_dir = os.path.join(
            evaluation_output_dir, "cnn_model_evaluation/")
        if os.path.isdir(output_dir):
            print(output_dir + " directory exists, delete it!")
            shutil.rmtree(output_dir)
        file_utils.makedirs(output_dir)
        print("Making output directory: " + output_dir)
        pred_points_dir = os.path.join(output_dir, 'pred_points/')
        file_utils.makedirs(pred_points_dir)

        total_frame_num = 0
        total_infeasible_num = 0
        for X, y, img_feature, coordinate_heading, message_timestamp_sec, \
                current_x, current_y, current_theta, current_v in tqdm(test_loader):
            X, y = cuda(X), cuda(y)
            pred = model(X)

            displacement_error, heading_error, v_error = \
                calculate_cnn_displacement_error(pred, y)
            displcement_errors.append(displacement_error)
            heading_errors.append(heading_error)
            v_errors.append(v_error)

            infeasible_num = calculate_kinematic_feasibility(
                pred, current_x, current_y, current_theta, current_v)
            total_infeasible_num += infeasible_num
            total_frame_num += img_feature.shape[0]

            visualize_cnn_result(output_renderer, img_feature, coordinate_heading,
                                 message_timestamp_sec, pred_points_dir, pred, y)

        average_displacement_error = 'average displacement error: {}.'.format(
            np.mean(displcement_errors))
        average_heading_error = 'average heading error: {}.'.format(
            np.mean(heading_errors))
        average_v_error = 'average speed error: {}.'.format(np.mean(v_errors))
        infeasibility_ratio = 'infeasibility_ratio: {}%.'.format(
            total_infeasible_num / total_frame_num * 100)

        with open(os.path.join(output_dir, "statistics.txt"), "w") as output_file:
            output_file.write(average_displacement_error + "\n")
            output_file.write(average_heading_error + "\n")
            output_file.write(average_v_error + "\n")
            output_file.write(infeasibility_ratio + "\n")

        print(average_displacement_error)
        print(average_heading_error)
        print(average_v_error)
        print(infeasibility_ratio)


def calculate_conv_rnn_displacement_error(pred, y):
    pred_points = pred[0]
    true_points = y[0]
    points_diff = pred_points - true_points
    pose_diff = points_diff[:, :, 0:2]
    heading_diff = points_diff[:, :, 2]
    v_diff = points_diff[:, :, 3]

    displacement_error = torch.mean(torch.sqrt(
        torch.sum(pose_diff ** 2, dim=-1))).item()
    heading_error = torch.mean(torch.abs(heading_diff)).item()
    v_error = torch.mean(torch.abs(v_diff)).item()
    return displacement_error, heading_error, v_error


def visualize_conv_rnn_result(renderer, img_feature, coordinate_heading,
                              message_timestamp_sec, pred_points_dir, explicit_memory_dir,
                              pos_dists_dir, pos_boxs_dir, pred, y):

    batched_pred_points = pred[0]
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
        true_point = y[0][i]
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

    batched_pred_pos_dists = pred[2]
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


def conv_rnn_model_evaluator(test_loader, model, renderer_config, evaluation_output_dir):
    with torch.no_grad():
        model.eval()

        displcement_errors = []
        heading_errors = []
        v_errors = []

        output_renderer = AgentPosesFutureImgRenderer(renderer_config)

        output_dir = os.path.join(
            evaluation_output_dir, "conv_rnn_model_evaluation/")
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
                calculate_conv_rnn_displacement_error(pred, y)
            displcement_errors.append(displacement_error)
            heading_errors.append(heading_error)
            v_errors.append(v_error)
            visualize_conv_rnn_result(
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


def evaluating(model_type, model_file, test_set_folder, gpu_idx, update_base_map,
               renderer_config_file, renderer_base_map_img_dir,
               renderer_base_map_data_dir, evaluation_output_dir, regions_list):
    # TODO(Jinyun): check performance
    cv.setNumThreads(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

    model = None
    test_dataset = None

    if update_base_map:
        ChauffeurNetFeatureGenerator.draw_base_map(regions_list,
                                                   renderer_config_file,
                                                   renderer_base_map_img_dir,
                                                   renderer_base_map_data_dir)
    renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
    renderer_config = proto_utils.get_pb_from_text_file(
        renderer_config_file, renderer_config)

    if model_type == 'cnn':
        model = TrajectoryImitationCNNFC(pred_horizon=10)
        test_dataset = TrajectoryImitationCNNFCDataset(test_set_folder,
                                                       regions_list,
                                                       renderer_config_file,
                                                       renderer_base_map_img_dir,
                                                       renderer_base_map_data_dir,
                                                       evaluate_mode=True)
    elif model_type == 'conv_rnn':
        model = TrajectoryImitationConvRNN(
            input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        # model = TrajectoryImitationDeeperConvRNN(
        #     input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        # model = TrajectoryImitationConvRNNUnetResnet18v1(
        #     input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        # model = TrajectoryImitationConvRNNUnetResnet18v2(
        #     input_img_size=[renderer_config.height, renderer_config.width], pred_horizon=10)
        test_dataset = TrajectoryImitationConvRNNDataset(test_set_folder,
                                                         regions_list,
                                                         renderer_config_file,
                                                         renderer_base_map_img_dir,
                                                         renderer_base_map_data_dir,
                                                         evaluate_mode=True)
    elif model_type == 'self_cnn_lstm' or model_type == 'self_cnn_lstm_aux':
        model = TrajectoryImitationSelfCNNLSTM(history_len=10, pred_horizon=10, embed_size=64,
                                               hidden_size=128)
        test_dataset = TrajectoryImitationSelfCNNLSTMDataset(test_set_folder,
                                                             regions_list,
                                                             renderer_config_file,
                                                             renderer_base_map_img_dir,
                                                             renderer_base_map_data_dir,
                                                             history_point_num=10,
                                                             ouput_point_num=10,
                                                             evaluate_mode=True)
    elif model_type == "unconstrained_cnn_lstm":
        model = TrajectoryImitationUnconstrainedCNNLSTM(pred_horizon=10)
        test_dataset = TrajectoryImitationCNNLSTMDataset(test_set_folder,
                                                         regions_list,
                                                         renderer_config_file,
                                                         renderer_base_map_img_dir,
                                                         renderer_base_map_data_dir,
                                                         ouput_point_num=10,
                                                         evaluate_mode=True)

    elif model_type == "kinematic_cnn_lstm":
        model = TrajectoryImitationKinematicConstrainedCNNLSTM(pred_horizon=10,
                                                               max_abs_steering_angle=0.52,
                                                               max_acceleration=2,
                                                               max_deceleration=-4,
                                                               wheel_base=2.8448,
                                                               delta_t=0.2)
        test_dataset = TrajectoryImitationCNNLSTMDataset(test_set_folder,
                                                         regions_list,
                                                         renderer_config_file,
                                                         renderer_base_map_img_dir,
                                                         renderer_base_map_data_dir,
                                                         ouput_point_num=10,
                                                         evaluate_mode=True)
    else:
        logging.info('model {} is not implemnted'.format(model_type))
        exit()

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last=True)

    model_state_dict = torch.load(model_file)
    model.load_state_dict(model_state_dict)

    # CUDA setup:
    if torch.cuda.is_available():
        print("Using CUDA to speed up training.")
        model.cuda()
    else:
        print("Not using CUDA.")

    if model_type == 'cnn' or model_type == 'self_cnn_lstm' or model_type == 'self_cnn_lstm_aux'\
            or model_type == "unconstrained_cnn_lstm" or model_type == "kinematic_cnn_lstm":
        cnn_model_evaluator(test_loader, model,
                            renderer_config_file, evaluation_output_dir)
    elif model_type == 'conv_rnn':
        conv_rnn_model_evaluator(test_loader, model,
                                 renderer_config_file, evaluation_output_dir)
    else:
        logging.info('model {} is not implemnted'.format(model_type))
        exit()


def main(argv):
    gflag = flags.FLAGS
    model_type = gflag.model_type
    model_file = gflag.model_file
    test_set_dir = gflag.test_set_dir
    regions_list = gflag.regions_list
    gpu_idx = gflag.gpu_idx
    update_base_map = gflag.update_base_map
    renderer_config_file = gflag.renderer_config_file
    renderer_base_map_img_dir = gflag.renderer_base_map_img_dir
    renderer_base_map_data_dir = gflag.renderer_base_map_data_dir
    evaluation_output_dir = gflag.evaluation_output_dir

    evaluating(model_type,
               model_file,
               test_set_dir,
               gpu_idx,
               update_base_map,
               renderer_config_file,
               renderer_base_map_img_dir,
               renderer_base_map_data_dir,
               evaluation_output_dir,
               regions_list)


if __name__ == "__main__":
    app.run(main)
