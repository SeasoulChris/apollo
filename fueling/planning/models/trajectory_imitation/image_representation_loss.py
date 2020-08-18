import torch
import torch.nn as nn

import fueling.common.logging as logging

'''
========================================================================
Loss definition
========================================================================
'''


class TrajectoryImitationConvRNNLoss():
    def __init__(self,
                 pos_dist_loss_weight=1,
                 box_loss_weight=1,
                 pos_reg_loss_weight=1):
        self.pos_dist_loss_weight = pos_dist_loss_weight
        self.box_loss_weight = box_loss_weight
        self.pos_reg_loss_weight = pos_reg_loss_weight

    def loss_fn(self, y_pred, y_true):
        batch_size = y_pred[0].shape[0]
        pred_pos_dists = y_pred[0].view(batch_size, -1)
        pred_boxs = y_pred[1].view(batch_size, -1)
        pred_points = y_pred[2].view(batch_size, -1)
        true_pos_dists = y_true[0].view(batch_size, -1)
        true_boxs = y_true[1].view(batch_size, -1)
        true_points = y_true[2].view(batch_size, -1)

        pos_dist_loss = nn.BCELoss()(pred_pos_dists, true_pos_dists)
        box_loss = nn.BCELoss()(pred_boxs, true_boxs)
        pos_reg_loss = nn.L1Loss()(pred_points, true_points)

        weighted_pos_dist_loss = self.pos_dist_loss_weight * pos_dist_loss
        weighted_box_loss = self.box_loss_weight * box_loss
        weighted_pos_reg_loss = self.pos_reg_loss_weight * pos_reg_loss

        logging.info("pos_dist_loss is {}, box_loss is {}, pos_reg_loss is {}".
                     format(
                         pos_dist_loss,
                         box_loss,
                         pos_reg_loss))
        logging.info("weighted_pos_dist_loss is {}, weighted_box_loss is {},"
                     "weighted_pos_reg_loss is {}".
                     format(
                         weighted_pos_dist_loss,
                         weighted_box_loss,
                         weighted_pos_reg_loss))

        return weighted_pos_dist_loss + \
            weighted_box_loss + \
            weighted_pos_reg_loss

    def loss_info(self, y_pred, y_true):
        pred_points = y_pred[2]
        true_points = y_true[2]
        points_diff = pred_points - true_points
        # First two elements are assumed to be x and y position
        pose_diff = points_diff[:, :, 0:2]
        out = torch.sqrt(torch.sum(pose_diff ** 2, dim=-1))
        out = torch.mean(out)
        return out


class TrajectoryImitationConvRNNWithEnvLoss():
    def __init__(self,
                 pos_dist_loss_weight=1,
                 box_loss_weight=1,
                 pos_reg_loss_weight=1,
                 collision_loss_weight=1,
                 offroad_loss_weight=1,
                 onrouting_loss_weight=1,
                 imitation_dropout=False):
        self.pos_dist_loss_weight = pos_dist_loss_weight
        self.box_loss_weight = box_loss_weight
        self.pos_reg_loss_weight = pos_reg_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.offroad_loss_weight = offroad_loss_weight
        self.onrouting_loss_weight = onrouting_loss_weight
        self.imitation_dropout = imitation_dropout

    def loss_fn(self, y_pred, y_true):
        batch_size = y_pred[0].shape[0]
        pred_pos_dists = y_pred[0].view(batch_size, -1)
        pred_boxs = y_pred[1].view(batch_size, -1)
        pred_points = y_pred[2].view(batch_size, -1)
        true_pos_dists = y_true[0].view(batch_size, -1)
        true_boxs = y_true[1].view(batch_size, -1)
        true_points = y_true[2].view(batch_size, -1)
        true_pred_obs = y_true[3].view(batch_size, -1)
        true_offroad_mask = y_true[4].view(batch_size, -1)
        true_onrouting_mask = y_true[5].view(batch_size, -1)

        pos_dist_loss = nn.BCELoss()(pred_pos_dists, true_pos_dists)
        box_loss = nn.BCELoss()(pred_boxs, true_boxs)
        pos_reg_loss = nn.L1Loss()(pred_points, true_points)
        collision_loss = torch.mean(pred_boxs * true_pred_obs)
        offroad_loss = torch.mean(pred_boxs * true_offroad_mask)
        onrouting_loss = torch.mean(pred_boxs * true_onrouting_mask)

        weighted_pos_dist_loss = self.pos_dist_loss_weight * pos_dist_loss
        weighted_box_loss = self.box_loss_weight * box_loss
        weighted_pos_reg_loss = self.pos_reg_loss_weight * pos_reg_loss
        weighted_collision_loss = self.collision_loss_weight * collision_loss
        weighted_offroad_loss = self.offroad_loss_weight * offroad_loss
        weighted_onrouting_loss = self.onrouting_loss_weight * onrouting_loss

        logging.info("pos_dist_loss is {}, box_loss is {}, pos_reg_loss is {},"
                     "collision_loss is {}, offroad_loss is {}, "
                     "onrouting_loss is {}".
                     format(
                         pos_dist_loss,
                         box_loss,
                         pos_reg_loss,
                         collision_loss,
                         offroad_loss,
                         onrouting_loss
                     ))
        logging.info("weighted_pos_dist_loss is {}, weighted_box_loss is {},"
                     "weighted_pos_reg_loss is {}, weighted_collision_loss is {}, "
                     "weighted_offroad_loss is {}, weighted_onrouting_loss is {}".
                     format(
                         weighted_pos_dist_loss,
                         weighted_box_loss,
                         weighted_pos_reg_loss,
                         weighted_collision_loss,
                         weighted_offroad_loss,
                         weighted_onrouting_loss))

        return (0 if self.imitation_dropout and torch.rand(1) > 0.5 else 1) * \
            (weighted_pos_dist_loss
             + weighted_box_loss
             + weighted_pos_reg_loss) + \
            weighted_collision_loss + \
            weighted_offroad_loss + \
            weighted_onrouting_loss

    def loss_info(self, y_pred, y_true):
        pred_points = y_pred[2]
        true_points = y_true[2]
        points_diff = pred_points - true_points
        # First two elements are assumed to be x and y position
        pose_diff = points_diff[:, :, 0:2]
        out = torch.sqrt(torch.sum(pose_diff ** 2, dim=-1))
        out = torch.mean(out)
        return out


class TrajectoryImitationSelfCNNLSTMWithRasterizerEnvLoss():
    def __init__(self,
                 box_loss_weight=1,
                 pos_reg_loss_weight=1,
                 collision_loss_weight=1,
                 offroad_loss_weight=1,
                 onrouting_loss_weight=1,
                 imitation_dropout=False):
        self.box_loss_weight = box_loss_weight
        self.pos_reg_loss_weight = pos_reg_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.offroad_loss_weight = offroad_loss_weight
        self.onrouting_loss_weight = onrouting_loss_weight
        self.imitation_dropout = imitation_dropout

    def loss_fn(self, y_pred, y_true):
        batch_size = y_pred[0].shape[0]
        pred_boxs = y_pred[0].view(batch_size, -1)
        pred_points = y_pred[1].view(batch_size, -1)
        true_boxs = y_true[0].view(batch_size, -1)
        true_points = y_true[1].view(batch_size, -1)
        true_pred_obs = y_true[2].view(batch_size, -1)
        true_offroad_mask = y_true[3].view(batch_size, -1)
        true_onrouting_mask = y_true[4].view(batch_size, -1)

        box_loss = nn.BCELoss()(pred_boxs, true_boxs)
        pos_reg_loss = nn.L1Loss()(pred_points, true_points)
        collision_loss = torch.mean(pred_boxs * true_pred_obs)
        offroad_loss = torch.mean(pred_boxs * true_offroad_mask)
        onrouting_loss = torch.mean(pred_boxs * true_onrouting_mask)

        weighted_box_loss = self.box_loss_weight * box_loss
        weighted_pos_reg_loss = self.pos_reg_loss_weight * pos_reg_loss
        weighted_collision_loss = self.collision_loss_weight * collision_loss
        weighted_offroad_loss = self.offroad_loss_weight * offroad_loss
        weighted_onrouting_loss = self.onrouting_loss_weight * onrouting_loss

        logging.info("box_loss is {}, pos_reg_loss is {},"
                     "collision_loss is {}, offroad_loss is {},"
                     "onrouting_loss is {}".
                     format(
                         box_loss,
                         pos_reg_loss,
                         collision_loss,
                         offroad_loss,
                         onrouting_loss
                     ))
        logging.info("weighted_box_loss is {},weighted_pos_reg_loss is {},"
                     "weighted_collision_loss is {}, "
                     "weighted_offroad_loss is {}, "
                     "weighted_onrouting_loss is {}".
                     format(
                         weighted_box_loss,
                         weighted_pos_reg_loss,
                         weighted_collision_loss,
                         weighted_offroad_loss,
                         weighted_onrouting_loss))

        return (0 if self.imitation_dropout and torch.rand(1) > 0.5 else 1) * \
            (weighted_box_loss
             + weighted_pos_reg_loss) + \
            weighted_collision_loss + \
            weighted_offroad_loss + \
            weighted_onrouting_loss

    def loss_info(self, y_pred, y_true):
        pred_points = y_pred[1]
        true_points = y_true[1]
        points_diff = pred_points - true_points
        # First two elements are assumed to be x and y position
        pose_diff = points_diff[:, :, 0:2]
        out = torch.sqrt(torch.sum(pose_diff ** 2, dim=-1))
        out = torch.mean(out)
        return out
