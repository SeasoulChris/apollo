import glob
import os

import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

from fueling.learning.network_utils import *
from fueling.learning.train_utils import *
from fueling.learning.backbone import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class TrajectoryPredictionSingleLoss():
    def loss_fn(self, y_pred, y_true, epoch=None):
        loss_func = nn.MSELoss(reduction='none')
        n = y_pred.shape[0]
        topk = 1
        y_true_tensor = torch.cat(y_true)
        loss = torch.sum(torch.sum(loss_func(y_pred, y_true_tensor), 2), 1)
        topk = 1
        if epoch is not None and epoch > 70:
            topk = 0.7
            print('change topk to {} for epoch {} '.format(topk, epoch))

        if topk == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(max(topk * loss.size()[0], 1)))
            return torch.mean(valid_loss)

    def loss_info(self, y_pred, y_true):
        n = y_pred.shape[0]
        out = y_pred - torch.cat(y_true)
        out = torch.sqrt(torch.sum(out ** 2, 2))
        out = torch.mean(out)
        return out


@torch.jit.script
def roi_pooling(rois, feature_map, scale_factor, stride):
    # type: (Tensor, Tensor, Tensor, Tensor) -> List[Tensor]
    out = []
    obj_num = rois.shape[0]
    for i in range(int(obj_num)):
        maxval = ((torch.max(rois[i, :], 0)[0]) * stride * scale_factor).int()
        minval = ((torch.min(rois[i, :], 0)[0]) * stride * scale_factor).int()
        print(maxval)
        print(minval)
        (xval, yval) = clip_boxes(torch.stack((minval[0], maxval[0])).int(),
                                  torch.stack((minval[1], maxval[1])).int(),
                                  torch.tensor([feature_map.shape[2], feature_map.shape[3]]).int())

        (xmin, xmax) = (xval[0], xval[1])
        (ymin, ymax) = (yval[0], yval[1])
        # out should be 1x512xfeaturemap
        debug = 1
        if debug:
            print('xmax xmin ymax ymin {} {} {} {}'.format(xmax, xmin, ymax, ymin))
        if (xmax - xmin) >= 1 and (ymax - ymin) >= 1:
            out.append(feature_map[:, :, xmin:xmax + 1, ymin:ymax + 1])
        else:
            print('invalid xmax {} xmin {} ymax {} ymin {}'.format(xmax, xmin, ymax, ymin))
            out.append(torch.zeros(1, 50).cuda())
    return out


# Trajectory Prediction Single Image Model
class TrajectoryPredictionSingle(nn.Module):
    def __init__(self, pred_len, observation_len):
        super(TrajectoryPredictionSingle, self).__init__()
        self.pred_len = pred_len
        self.observation_len = observation_len

        self.disp_embed_x = torch.nn.Sequential(
            nn.Conv1d(1, 64, 3),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
        )

        self.disp_embed_y = torch.nn.Sequential(
            nn.Conv1d(1, 64, 3),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
        )

        self.resnet = resbackbone1()
        self.adaptive_avg_pool = torch.nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512, 50),
            nn.ReLU(),
        )
        self.predict_x = nn.Linear(50 * 2, pred_len)
        self.predict_y = nn.Linear(50 * 2, pred_len)

    def forward(self, X):
        pred_traj = []
        batchsize = len(X)

        for i in range(0, batchsize):
            data = X[i]
            img = data[0]
            channel, height, width = img.shape
            obs_pos_rel = data[1]
            ref_pos = data[2]
            rois = data[3].data.cpu()
            scale_factor = data[4].data.cpu()

            obj_num = obs_pos_rel.shape[0]
            hist_size = obs_pos_rel.shape[1]
            channel_size = 1
            pred_x = self.disp_embed_x(obs_pos_rel[:, :, 0].view(obj_num, channel_size, hist_size))
            pred_y = self.disp_embed_y(obs_pos_rel[:, :, 1].view(obj_num, channel_size, hist_size))
            feature_map = self.resnet(img[None, :])
            # original height / feature map height
            stride = float(feature_map.shape[2]) / float(height)
            out = roi_pooling(rois, feature_map, scale_factor, torch.tensor(stride))
            out = [self.adaptive_avg_pool(x) for x in out]
            output_concat = torch.cat(out, 0)
            x_concat = torch.cat((pred_x, output_concat), 1)
            y_concat = torch.cat((pred_y, output_concat), 1)
            x_predict = self.predict_x(x_concat)
            y_predict = self.predict_y(y_concat)
            pred_traj.append(
                torch.cat(
                    (x_predict.view(
                        obj_num, 10, 1), y_predict.view(
                        obj_num, 10, 1)), 2))

        pred_traj_tensor = torch.cat(pred_traj)
        return pred_traj_tensor
