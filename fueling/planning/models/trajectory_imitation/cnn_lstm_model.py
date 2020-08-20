import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from fueling.learning.network_utils import generate_lstm

'''
========================================================================
Model definition
========================================================================
'''


class TrajectoryImitationSelfCNNLSTM(nn.Module):
    def __init__(self, history_len, pred_horizon, embed_size=64,
                 hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True):
        super(TrajectoryImitationSelfCNNLSTM, self).__init__()
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.history_len = history_len
        self.pred_horizon = pred_horizon

        self.embedding_fc_layer = torch.nn.Sequential(
            nn.Linear(4, embed_size),
            nn.ReLU(),
        )

        self.h0, self.c0, self.lstm = generate_lstm(embed_size, hidden_size)

        self.output_fc_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size, 4),
        )

    def forward(self, X):
        img_feature, hist_points, hist_points_step = X
        batch_size = img_feature.size(0)
        # manually add the unsqueeze before repeat to avoid onnx to tensorRT parsing error
        h0 = self.h0.unsqueeze(0)
        c0 = self.c0.unsqueeze(0)
        ht, ct = h0.repeat(1, batch_size, 1),\
            c0.repeat(1, batch_size, 1)

        img_embedding = self.cnn(
            self.compression_cnn_layer(img_feature)).view(batch_size, -1)
        pred_traj = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)

        for t in range(1, self.history_len + self.pred_horizon):
            if t < self.history_len:
                cur_pose_step = hist_points_step[:, t, :].float()
                cur_pose = hist_points[:, t, :].float()
            else:
                pred_input = torch.cat(
                    (ht.view(batch_size, -1), img_embedding), 1)
                cur_pose_step = self.output_fc_layer(
                    pred_input).float().clone()
                cur_pose = cur_pose + cur_pose_step
                pred_traj = torch.cat(
                    (pred_traj, cur_pose.clone().unsqueeze(1)), dim=1)

            disp_embedding = self.embedding_fc_layer(
                cur_pose_step.clone()).view(batch_size, 1, -1)

            _, (ht, ct) = self.lstm(disp_embedding, (ht, ct))

        return pred_traj[:, 1:, :]


class TrajectoryImitationUnconstrainedCNNLSTM(nn.Module):
    def __init__(self, pred_horizon, embed_size=64,
                 hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True):
        super(TrajectoryImitationUnconstrainedCNNLSTM, self).__init__()
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.pred_horizon = pred_horizon

        self.h0_fc_layer = torch.nn.Sequential(
            nn.Linear(3 + self.cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.c0 = torch.zeros(1, 1, hidden_size)
        self.c0 = nn.Parameter(self.c0, requires_grad=True)

        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers=1, batch_first=True)

        self.embedding_fc_layer = torch.nn.Sequential(
            nn.Linear(4, embed_size),
            nn.ReLU()
        )

        self.output_fc_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size + 3, 4),
        )

    def forward(self, X):
        img_feature, current_v, current_a, current_curvature = X
        batch_size = img_feature.size(0)
        ct = self.c0.repeat(1, batch_size, 1)
        img_embedding = self.cnn(
            self.compression_cnn_layer(img_feature)).view(batch_size, -1)
        concatenated_states = torch.cat(
            (img_embedding, current_v, current_a, current_curvature), dim=1)
        ht = self.h0_fc_layer(concatenated_states).unsqueeze(0)
        path_point_input = torch.zeros(
            (batch_size, 3), device=img_feature.device)
        current_pose = torch.cat(
            (path_point_input, current_v.clone()), dim=1)

        pred_traj = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)
        for t in range(self.pred_horizon):
            current_pose_step = self.output_fc_layer(
                torch.cat((ht.view(batch_size, -1), concatenated_states), dim=1))

            current_pose = current_pose + current_pose_step

            pred_traj = torch.cat(
                (pred_traj, current_pose.clone().unsqueeze(1)), dim=1)

            states_input = self.embedding_fc_layer(
                current_pose_step.clone()).view(batch_size, 1, -1)

            _, (ht, ct) = self.lstm(states_input, (ht, ct))

        return pred_traj[:, 1:, :]


class TrajectoryImitationKinematicConstrainedCNNLSTM(nn.Module):
    def __init__(self, pred_horizon, max_abs_steering_angle, max_acceleration,
                 max_deceleration, delta_t, wheel_base, embed_size=64,
                 hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True):
        super(TrajectoryImitationKinematicConstrainedCNNLSTM, self).__init__()
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.pred_horizon = pred_horizon

        self.h0_fc_layer = torch.nn.Sequential(
            nn.Linear(3 + self.cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.c0 = torch.zeros(1, 1, hidden_size)
        self.c0 = nn.Parameter(self.c0, requires_grad=True)

        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers=1, batch_first=True)

        self.embedding_fc_layer = torch.nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU()
        )

        self.output_fc_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size + 3, 2),
        )

        self.max_abs_steering_angle = max_abs_steering_angle
        self.acceleration_range = max_acceleration - max_deceleration
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.delta_t = delta_t
        self.wheel_base = wheel_base
        # TODO(Jinyun): it's not accurate as xy pose is given as rear center points
        self.lf = wheel_base / 2
        self.lr = wheel_base / 2

    def kinematic_constraints_layer(self, current_action):
        steering = current_action[:, 0]
        a = current_action[:, 1]
        constrained_steering = torch.tanh(
            steering) * self.max_abs_steering_angle
        constrained_a = torch.tanh(a) * 0.5 * self.acceleration_range
        + (self.max_acceleration - 0.5 * self.acceleration_range)
        return torch.cat((constrained_steering.clone().unsqueeze(1),
                          constrained_a.clone().unsqueeze(1)), dim=1)

    def rear_kinematic_model_layer(self, current_states, current_action):
        '''
        current_states: x_0, y_0, phi_0, v_0
        current_action: steering_0, a_0
        return predicted_states
        '''
        x_0 = current_states[:, 0]
        y_0 = current_states[:, 1]
        phi_0 = current_states[:, 2]
        v_0 = current_states[:, 3]
        steering_0 = current_action[:, 0]
        a_0 = current_action[:, 1]

        delta_v = a_0
        delta_phi = (v_0 + 0.5 * delta_v) / \
            self.wheel_base * torch.tan(steering_0)
        delta_x = (v_0 + 0.5 * delta_v) * torch.cos(phi_0 + 0.5 * delta_phi)
        delta_y = (v_0 + 0.5 * delta_v) * torch.sin(phi_0 + 0.5 * delta_phi)

        # delta_v = a_0
        # delta_phi = v_0 / self.wheel_base * torch.tan(steering_0)
        # delta_x = v_0 * torch.cos(phi_0)
        # delta_y = v_0 * torch.sin(phi_0)

        x_1 = delta_x * self.delta_t + x_0
        y_1 = delta_y * self.delta_t + y_0
        phi_1 = delta_phi * self.delta_t + phi_0
        v_1 = delta_v * self.delta_t + v_0

        return torch.cat((x_1.unsqueeze(1),
                          y_1.unsqueeze(1),
                          phi_1.unsqueeze(1),
                          v_1.unsqueeze(1)), dim=1)

    def center_kinematic_model_layer(self, current_states, current_action):
        '''
        current_states: x_0, y_0, phi_0, v_0
        current_action: steering_0, a_0
        return predicted_states
        '''
        x_0 = current_states[:, 0]
        y_0 = current_states[:, 1]
        phi_0 = current_states[:, 2]
        v_0 = current_states[:, 3]
        steering_0 = current_action[:, 0]
        a_0 = current_action[:, 1]

        beta = torch.atan(self.lr / (self.lr + self.lf)
                          * torch.tan(steering_0))
        delta_x = v_0 * torch.cos(phi_0 + beta)
        delta_y = v_0 * torch.sin(phi_0 + beta)
        delta_phi = v_0 / self.lr * torch.sin(beta)
        delta_v = a_0

        x_1 = delta_x * self.delta_t + x_0
        y_1 = delta_y * self.delta_t + y_0
        phi_1 = delta_phi * self.delta_t + phi_0
        v_1 = delta_v * self.delta_t + v_0

        return torch.cat((x_1.unsqueeze(1),
                          y_1.unsqueeze(1),
                          phi_1.unsqueeze(1),
                          v_1.unsqueeze(1)), dim=1)

    def forward(self, X):
        img_feature, current_v, current_a, current_curvature = X
        batch_size = img_feature.size(0)
        ct = self.c0.repeat(1, batch_size, 1)
        img_embedding = self.cnn(
            self.compression_cnn_layer(img_feature)).view(batch_size, -1)
        concatenated_states = torch.cat(
            (img_embedding, current_v, current_a, current_curvature), dim=1)
        ht = self.h0_fc_layer(concatenated_states).unsqueeze(0)
        path_point_input = torch.zeros(
            (batch_size, 3), device=img_feature.device)
        current_states = torch.cat(
            (path_point_input, current_v.clone()), dim=1)

        # model output as [steering, a]
        pred_traj = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)
        pred_traj = torch.cat(
            (pred_traj, current_states.clone().unsqueeze(1)), dim=1)
        for t in range(self.pred_horizon):
            current_action = self.output_fc_layer(
                torch.cat((ht.view(batch_size, -1), concatenated_states), dim=1))

            current_action = self.kinematic_constraints_layer(
                current_action)

            previous_states = pred_traj[:, -1, :].clone()

            current_states = self.rear_kinematic_model_layer(
                previous_states, current_action)

            pred_traj = torch.cat(
                (pred_traj, current_states.clone().unsqueeze(1)), dim=1)

            states_input = self.embedding_fc_layer(
                current_action.clone()).view(batch_size, 1, -1)

            _, (ht, ct) = self.lstm(states_input, (ht, ct))

        return pred_traj[:, 2:, :]


def rasterize_vehicle_box(pred_traj_x, pred_traj_y, box_heading,
                          heading_offset,
                          initial_box_x_idx,
                          initial_box_y_idx,
                          center_shift_distance,
                          img_resolution,
                          half_box_length, half_box_width, idx_mesh):
    rear_to_initial_box_x_delta = (torch.cos(heading_offset) * (-pred_traj_y)
                                   - torch.sin(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)
    rear_to_initial_box_y_delta = (torch.sin(heading_offset) * (-pred_traj_y)
                                   + torch.cos(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)

    center_to_rear_x_delta = torch.cos(
        box_heading) * center_shift_distance
    center_to_rear_y_delta = torch.sin(
        box_heading) * center_shift_distance

    center_to_initial_box_x_delta = center_to_rear_x_delta + rear_to_initial_box_x_delta
    center_to_initial_box_y_delta = center_to_rear_y_delta + rear_to_initial_box_y_delta

    mid_center_x_idx = (
        initial_box_x_idx + torch.div(center_to_initial_box_x_delta,
                                      img_resolution))
    mid_center_y_idx = (
        initial_box_y_idx + torch.div(center_to_initial_box_y_delta,
                                      img_resolution))

    delta_x = idx_mesh[:, :, :, 1] - mid_center_x_idx
    delta_y = idx_mesh[:, :, :, 0] - mid_center_y_idx
    abs_transformed_delta_x = torch.abs(torch.cos(-box_heading) * delta_x
                                        - torch.sin(-box_heading) * delta_y)
    abs_transformed_delta_y = torch.abs(torch.sin(-box_heading) * delta_x
                                        + torch.cos(-box_heading) * delta_y)
    return torch.logical_and(abs_transformed_delta_x < half_box_length,
                             abs_transformed_delta_y < half_box_width).float()


def rasterize_vehicle_three_circles_guassian(pred_traj_x,
                                             pred_traj_y,
                                             box_heading,
                                             heading_offset,
                                             initial_box_x_idx,
                                             initial_box_y_idx,
                                             center_shift_distance,
                                             front_shift_distance,
                                             img_resolution,
                                             theta,
                                             sigma_x,
                                             sigma_y,
                                             idx_mesh,
                                             front_center_guassian,
                                             mid_center_guassian,
                                             rear_center_guassian):
    rear_to_initial_box_x_delta = (torch.cos(heading_offset) * (-pred_traj_y)
                                   - torch.sin(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)
    rear_to_initial_box_y_delta = (torch.sin(heading_offset) * (-pred_traj_y)
                                   + torch.cos(heading_offset) * (-pred_traj_x)).\
        unsqueeze(1).unsqueeze(2)

    center_to_rear_x_delta = torch.cos(
        box_heading) * center_shift_distance
    center_to_rear_y_delta = torch.sin(
        box_heading) * center_shift_distance

    center_to_initial_box_x_delta = center_to_rear_x_delta + rear_to_initial_box_x_delta
    center_to_initial_box_y_delta = center_to_rear_y_delta + rear_to_initial_box_y_delta

    rear_center_x_idx = (
        initial_box_x_idx + torch.div(rear_to_initial_box_x_delta,
                                      img_resolution))
    rear_center_y_idx = (
        initial_box_y_idx + torch.div(rear_to_initial_box_y_delta,
                                      img_resolution))

    mid_center_x_idx = (
        initial_box_x_idx + torch.div(center_to_initial_box_x_delta,
                                      img_resolution))
    mid_center_y_idx = (
        initial_box_y_idx + torch.div(center_to_initial_box_y_delta,
                                      img_resolution))

    center_to_front_x_delta = torch.cos(
        box_heading) * front_shift_distance
    center_to_front_y_delta = torch.sin(
        box_heading) * front_shift_distance

    front_to_initial_box_x_delta = center_to_front_x_delta + rear_to_initial_box_x_delta
    front_to_initial_box_y_delta = center_to_front_y_delta + rear_to_initial_box_y_delta

    front_center_x_idx = (
        initial_box_x_idx + torch.div(front_to_initial_box_x_delta,
                                      img_resolution))
    front_center_y_idx = (
        initial_box_y_idx + torch.div(front_to_initial_box_y_delta,
                                      img_resolution))

    a = torch.pow(torch.cos(theta), 2) / (2 * torch.pow(sigma_x, 2)) + \
        torch.pow(torch.sin(theta), 2) / (2 * torch.pow(sigma_y, 2))
    b = torch.sin(2 * theta) / (4 * torch.pow(sigma_x, 2)) + \
        torch.sin(2 * theta) / (4 * torch.pow(sigma_y, 2))
    c = torch.pow(torch.sin(theta), 2) / (2 * torch.pow(sigma_x, 2)) + \
        torch.pow(torch.cos(theta), 2) / (2 * torch.pow(sigma_y, 2))
    x_coords = idx_mesh[:, :, :, 1]
    y_coords = idx_mesh[:, :, :, 0]
    front_center_guassian = torch.exp(-(a * torch.pow((x_coords - front_center_x_idx), 2)
                                        + 2 * b
                                        * (x_coords - front_center_x_idx)
                                        * (y_coords - front_center_y_idx)
                                        + c * torch.pow((y_coords - front_center_y_idx), 2)))

    mid_center_guassian = torch.exp(-(a * torch.pow((x_coords - mid_center_x_idx), 2)
                                      + 2 * b * (x_coords - mid_center_x_idx)
                                      * (y_coords - mid_center_y_idx)
                                      + c * torch.pow((y_coords - mid_center_y_idx), 2)))

    rear_center_guassian = torch.exp(-(a * torch.pow((x_coords - rear_center_x_idx), 2)
                                       + 2 * b * (x_coords - rear_center_x_idx)
                                       * (y_coords - rear_center_y_idx)
                                       + c * torch.pow((y_coords - rear_center_y_idx), 2)))
    max_prob = 1.0010
    return (front_center_guassian + mid_center_guassian + rear_center_guassian) / max_prob


class TrajectoryImitationSelfCNNLSTMWithRasterizer(nn.Module):
    def __init__(self, history_len, pred_horizon, input_img_size, img_resolution,
                 initial_box_x_idx, initial_box_y_idx,
                 vehicle_front_edge_to_center, vehicle_back_edge_to_center, vehicle_width,
                 embed_size=64, hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True):
        super(TrajectoryImitationSelfCNNLSTMWithRasterizer, self).__init__()
        self.compression_cnn_layer = nn.Conv2d(12, 3, 3, padding=1)
        self.cnn = cnn_net(pretrained=pretrained)
        self.cnn_out_size = 1000
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.history_len = history_len
        self.pred_horizon = pred_horizon
        self.input_img_size_h = input_img_size[0]
        self.input_img_size_w = input_img_size[1]

        self.embedding_fc_layer = torch.nn.Sequential(
            nn.Linear(4, embed_size),
            nn.ReLU(),
        )

        self.h0, self.c0, self.lstm = generate_lstm(embed_size, hidden_size)

        self.output_fc_layer = torch.nn.Sequential(
            nn.Linear(hidden_size + self.cnn_out_size, 4),
        )

        self.img_resolution = img_resolution
        x_mesh = torch.Tensor(np.arange(self.input_img_size_w))
        y_mesh = torch.Tensor(np.arange(self.input_img_size_h))
        x_mesh, y_mesh = torch.meshgrid(x_mesh, y_mesh)
        self.idx_mesh = torch.stack((x_mesh, y_mesh), dim=2)
        self.initial_box_x_idx = initial_box_x_idx
        self.initial_box_y_idx = initial_box_y_idx
        half_box_length = (
            vehicle_front_edge_to_center + vehicle_back_edge_to_center) / 2
        self.center_shift_distance = half_box_length - vehicle_back_edge_to_center
        self.front_shift_distance = half_box_length * 2 - 2 * vehicle_back_edge_to_center
        # use geometry algorithm to subdifferentiably draw the box
        # self.half_box_length = ( vehicle_front_edge_to_center
        #                       + vehicle_back_edge_to_center) / 2
        # self.half_box_width = vehicle_width

    def forward(self, X):
        img_feature, hist_points, hist_points_step, ego_rendering_heading = X
        ego_rendering_heading = ego_rendering_heading.squeeze()
        batch_size = img_feature.size(0)
        # manually add the unsqueeze before repeat to avoid onnx to tensorRT parsing error
        h0 = self.h0.unsqueeze(0)
        c0 = self.c0.unsqueeze(0)
        ht, ct = h0.repeat(1, batch_size, 1),\
            c0.repeat(1, batch_size, 1)
        idx_mesh = self.idx_mesh.repeat(
            batch_size, 1, 1, 1).to(img_feature.device)

        img_embedding = self.cnn(
            self.compression_cnn_layer(img_feature)).view(batch_size, -1)
        pred_traj = torch.zeros(
            (batch_size, 1, 4), device=img_feature.device)

        for t in range(1, self.history_len + self.pred_horizon):
            if t < self.history_len:
                cur_pose_step = hist_points_step[:, t, :].float()
                cur_pose = hist_points[:, t, :].float()
            else:
                pred_input = torch.cat(
                    (ht.view(batch_size, -1), img_embedding), 1)
                cur_pose_step = self.output_fc_layer(
                    pred_input).float().clone()
                cur_pose = cur_pose + cur_pose_step
                pred_traj = torch.cat(
                    (pred_traj, cur_pose.clone().unsqueeze(1)), dim=1)

            disp_embedding = self.embedding_fc_layer(
                cur_pose_step.clone()).view(batch_size, 1, -1)

            _, (ht, ct) = self.lstm(disp_embedding, (ht, ct))

        pred_boxs = torch.zeros(
            (batch_size, self.pred_horizon, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        for t in range(self.pred_horizon):
            box_heading = (-pred_traj[:, t, 2]
                           - ego_rendering_heading).unsqueeze(1).unsqueeze(2)
            heading_offset = -(ego_rendering_heading - np.pi / 2)

            front_center_guassian = torch.zeros(
                (batch_size, 1, self.input_img_size_h, self.input_img_size_w),
                requires_grad=True, device=img_feature.device)
            mid_center_guassian = torch.zeros(
                (batch_size, 1, self.input_img_size_h, self.input_img_size_w),
                requires_grad=True, device=img_feature.device)
            rear_center_guassian = torch.zeros(
                (batch_size, 1, self.input_img_size_h, self.input_img_size_w),
                requires_grad=True, device=img_feature.device)
            theta = torch.Tensor([0.0]).to(img_feature.device)
            sigma_x = torch.Tensor([1.8]).to(img_feature.device)
            sigma_y = torch.Tensor([1.8]).to(img_feature.device)
            pred_boxs[:, t, 0, :, :] = \
                rasterize_vehicle_three_circles_guassian(
                pred_traj[:, t, 0],
                pred_traj[:, t, 1],
                box_heading,
                heading_offset,
                self.initial_box_x_idx,
                self.initial_box_y_idx,
                self.center_shift_distance,
                self.front_shift_distance,
                self.img_resolution,
                theta,
                sigma_x,
                sigma_y,
                idx_mesh,
                front_center_guassian[:,
                                      0, :, :],
                mid_center_guassian[:,
                                    0, :, :],
                rear_center_guassian[:, 0, :, :],)

            # use geometry algorithm to subdifferentiably draw the box
            # pred_boxs[:, t, 0, :, :] = rasterize_vehicle_box(
            #     pred_traj[:, t, 0],
            #     pred_traj[:, t, 1],
            #     box_heading,
            #     heading_offset,
            #     self.initial_box_x_idx,
            #     self.initial_box_y_idx,
            #     self.center_shift_distance,
            #     self.img_resolution,
            #     self.half_box_length,
            #     self.half_box_width,
            #     idx_mesh)

        # for model export usage
        # return pred_traj[:, 1:, :]

        return pred_traj[:, 1:, :], pred_boxs


if __name__ == "__main__":
    # code snippet for model prining
    model = TrajectoryImitationSelfCNNLSTM(10, 10)

    total_param = 0
    for name, param in model.named_parameters():
        print(name)
        print(param.data.shape)
        print(param.data.view(-1).shape[0])
        total_param += param.data.view(-1).shape[0]
    print(total_param)

    for index, (name, child) in enumerate(model.named_children()):
        print(index)
        print(name)
        print(child)
