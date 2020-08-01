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


class TrajectoryImitationCNNLSTM(nn.Module):
    def __init__(self, history_len, pred_horizon, embed_size=64,
                 hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True):
        super(TrajectoryImitationCNNLSTM, self).__init__()
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


def rasterize_vehicle_box(box_center_x_idx, box_center_y_idx, box_heading,
                          half_box_length, half_box_width, idx_mesh):
    delta_x = idx_mesh[:, :, :, 1] - box_center_x_idx
    delta_y = idx_mesh[:, :, :, 0] - box_center_y_idx
    abs_transformed_delta_x = torch.abs(torch.cos(-box_heading) * delta_x
                                        - torch.sin(-box_heading) * delta_y)
    abs_transformed_delta_y = torch.abs(torch.sin(-box_heading) * delta_x
                                        + torch.cos(-box_heading) * delta_y)
    return torch.logical_and(abs_transformed_delta_x < half_box_length,
                             abs_transformed_delta_y < half_box_width).float()


def rasterize_vehicle_three_circles_guassian(front_center_x_idx, front_center_y_idx,
                                             mid_center_x_idx, mid_center_y_idx,
                                             rear_center_x_idx, rear_center_y_idx,
                                             front_center_guassian,
                                             mid_center_guassian,
                                             rear_center_guassian,
                                             theta,
                                             sigma_x,
                                             sigma_y,
                                             idx_mesh):
    a = torch.pow(torch.cos(theta), 2) / (2 * torch.pow(sigma_x, 2)) + \
        torch.pow(torch.sin(theta), 2) / (2 * torch.pow(sigma_y, 2))
    b = torch.sin(2 * theta) / (4 * torch.pow(sigma_x, 2)) + \
        torch.sin(2 * theta) / (4 * torch.pow(sigma_y, 2))
    c = torch.pow(torch.sin(theta), 2) / (2 * torch.pow(sigma_x, 2)) + \
        torch.pow(torch.cos(theta), 2) / (2 * torch.pow(sigma_y, 2))
    x_coords = idx_mesh[:, :, :, 1]
    y_coords = idx_mesh[:, :, :, 0]
    front_center_guassian = torch.exp(-(a * torch.pow((x_coords - front_center_x_idx), 2)
                                        + 2 * b * (x_coords - front_center_x_idx)
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


class TrajectoryImitationCNNLSTMWithAuxilaryEvaluationNet(nn.Module):
    def __init__(self, history_len, pred_horizon, input_img_size, img_resolution,
                 initial_box_x_idx, initial_box_y_idx,
                 vehicle_front_edge_to_center, vehicle_back_edge_to_center, vehicle_width,
                 embed_size=64, hidden_size=128, cnn_net=models.mobilenet_v2,
                 pretrained=True):
        super(TrajectoryImitationCNNLSTMWithAuxilaryEvaluationNet, self).__init__()
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

        # Using a fc layers to try to approximate the process of rasterization
        # self.box_pred_fc_layer = torch.nn.Sequential(
        #     nn.Linear(4 + self.cnn_out_size,
        #               self.input_img_size_h * self.input_img_size_w),
        # )

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

        # Using a fc layers to try to approximate the process of rasterization
        # pred_boxs = torch.zeros(
        #     (batch_size, 1, 1,
        #      self.input_img_size_h, self.input_img_size_w),
        #     device=img_feature.device)
        # for t in range(self.pred_horizon):
        #     pred_box = torch.sigmoid(self.box_pred_fc_layer(
        #         torch.cat((img_embedding, pred_traj[:, t, :]), dim=1)))
        #     pred_boxs = torch.cat((pred_boxs, pred_box.clone().view(
        #         batch_size, 1, 1, self.input_img_size_h, self.input_img_size_w)), dim=1)
        # return pred_boxs[:, 1:, :, :, :], pred_traj[:, 1:, :]

        pred_boxs = torch.zeros(
            (batch_size, self.pred_horizon, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        for t in range(self.pred_horizon):
            box_heading = (-pred_traj[:, t, 2]
                           - ego_rendering_heading).unsqueeze(1).unsqueeze(2)
            heading_offset = -(ego_rendering_heading - np.pi / 2)

            rear_to_initial_box_x_delta = (torch.cos(heading_offset) * (-pred_traj[:, t, 1])
                                           - torch.sin(heading_offset) * (-pred_traj[:, t, 0])).\
                unsqueeze(1).unsqueeze(2)
            rear_to_initial_box_y_delta = (torch.sin(heading_offset) * (-pred_traj[:, t, 1])
                                           + torch.cos(heading_offset) * (-pred_traj[:, t, 0])).\
                unsqueeze(1).unsqueeze(2)

            center_to_rear_x_delta = torch.cos(
                box_heading) * self.center_shift_distance
            center_to_rear_y_delta = torch.sin(
                box_heading) * self.center_shift_distance

            center_to_initial_box_x_delta = center_to_rear_x_delta + rear_to_initial_box_x_delta
            center_to_initial_box_y_delta = center_to_rear_y_delta + rear_to_initial_box_y_delta

            rear_center_x_idx = (
                self.initial_box_x_idx + torch.div(rear_to_initial_box_x_delta,
                                                   self.img_resolution))
            rear_center_y_idx = (
                self.initial_box_x_idx + torch.div(rear_to_initial_box_y_delta,
                                                   self.img_resolution))

            box_center_x_idx = (
                self.initial_box_x_idx + torch.div(center_to_initial_box_x_delta,
                                                   self.img_resolution))
            box_center_y_idx = (
                self.initial_box_y_idx + torch.div(center_to_initial_box_y_delta,
                                                   self.img_resolution))

            center_to_front_x_delta = torch.cos(
                box_heading) * self.front_shift_distance
            center_to_front_y_delta = torch.sin(
                box_heading) * self.front_shift_distance

            front_to_initial_box_x_delta = center_to_front_x_delta + rear_to_initial_box_x_delta
            front_to_initial_box_y_delta = center_to_front_y_delta + rear_to_initial_box_y_delta

            front_center_x_idx = (
                self.initial_box_x_idx + torch.div(front_to_initial_box_x_delta,
                                                   self.img_resolution))
            front_center_y_idx = (
                self.initial_box_x_idx + torch.div(front_to_initial_box_y_delta,
                                                   self.img_resolution))

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
            pred_boxs[:, t, 0, :, :] = rasterize_vehicle_three_circles_guassian(
                front_center_x_idx, front_center_y_idx,
                box_center_x_idx, box_center_y_idx,
                rear_center_x_idx, rear_center_y_idx,
                front_center_guassian[:, 0, :, :],
                mid_center_guassian[:, 0, :, :],
                rear_center_guassian[:, 0, :, :],
                theta,
                sigma_x,
                sigma_y,
                idx_mesh)

            # use geometry algorithm to subdifferentiably draw the box
            # pred_boxs[:, t, 0, :, :] = rasterize_vehicle_box(
            #     box_center_x_idx,
            #     box_center_y_idx,
            #     box_heading,
            #     self.half_box_length / self.img_resolution,
            #     self.half_box_width / self.img_resolution,
            #     idx_mesh)

        return pred_boxs, pred_traj[:, 1:, :]


if __name__ == "__main__":
    # code snippet for model prining
    model = TrajectoryImitationCNNLSTM(10, 10)

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
