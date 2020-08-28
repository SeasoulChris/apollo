import torch
import torch.nn as nn

'''
========================================================================
Loss definition
========================================================================
'''


class TrajectoryPointDisplacementMSELoss():
    def __init__(self, states_num=4):
        self.states_num = states_num

    def loss_fn(self, y_pred, y_true):
        loss_func = nn.MSELoss()
        losses = dict()
        losses["total_loss"] = loss_func(y_pred, y_true)
        return losses

    def loss_info(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        out = (y_pred - y_true).view(batch_size, -1, self.states_num)
        # First two elements are assumed to be x and y position
        pose_diff = out[:, :, 0:2]
        out = torch.mean(torch.sqrt(torch.sum(pose_diff ** 2, dim=-1)))
        return out


class TrajectoryPointDisplacementL1Loss():
    def __init__(self, states_num=4):
        self.states_num = states_num

    def loss_fn(self, y_pred, y_true):
        loss_func = nn.L1Loss()
        losses = dict()
        losses["total_loss"] = loss_func(y_pred, y_true)
        return losses

    def loss_info(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        out = (y_pred - y_true).view(batch_size, -1, self.states_num)
        # First two elements are assumed to be x and y position
        pose_diff = out[:, :, 0:2]
        out = torch.mean(torch.sqrt(torch.sum(pose_diff ** 2, dim=-1)))
        return out
