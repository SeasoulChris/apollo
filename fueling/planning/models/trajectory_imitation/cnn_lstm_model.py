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


class TrajectoryImitationCNNLSTMWithAuxilaryEvaluationNet(nn.Module):
    def __init__(self, history_len, pred_horizon, input_img_size, embed_size=64,
                 hidden_size=128, cnn_net=models.mobilenet_v2,
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

        self.box_pred_fc_layer = torch.nn.Sequential(
            nn.Linear(4 + self.cnn_out_size,
                      self.input_img_size_h * self.input_img_size_w),
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

        pred_boxs = torch.zeros(
            (batch_size, 1, 1,
             self.input_img_size_h, self.input_img_size_w),
            device=img_feature.device)
        for t in range(self.pred_horizon):
            pred_box = torch.sigmoid(self.box_pred_fc_layer(
                torch.cat((img_embedding, pred_traj[:, t, :]), dim=1)))
            pred_boxs = torch.cat((pred_boxs, pred_box.clone().view(
                batch_size, 1, 1, self.input_img_size_h, self.input_img_size_w)), dim=1)

        # return pred_traj[:, 1:, :]

        return pred_boxs[:, 1:, :, :, :], pred_traj[:, 1:, :]


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
