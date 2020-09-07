import torch
import torch.nn as nn

import fueling.common.logging as logging

'''
========================================================================
Loss definition
========================================================================
'''


def clamped_exp(x):
    return torch.exp(torch.clamp(x, max=50))


class ImageRepresentationLoss():
    def __init__(self,
                 pos_reg_loss_weight=1,
                 pos_dist_loss_weight=1,
                 box_loss_weight=1,
                 collision_loss_weight=1,
                 offroad_loss_weight=1,
                 onrouting_loss_weight=1,
                 imitation_dropout=False,
                 batchwise_focal_loss=False,
                 losswise_focal_loss=False,
                 focal_loss_gamma=1):
        self.pos_reg_loss_weight = pos_reg_loss_weight
        self.box_loss_weight = box_loss_weight
        self.pos_dist_loss_weight = pos_dist_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.offroad_loss_weight = offroad_loss_weight
        self.onrouting_loss_weight = onrouting_loss_weight
        self.imitation_dropout = imitation_dropout
        self.batchwise_focal_loss = batchwise_focal_loss
        self.losswise_focal_loss = losswise_focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        # TODO(Jinyun): use logsoftmax for more numerical stability
        self.kepsilon = 1e-6
        self.is_synthesized_weight = 0.1
        self.is_turning_weight = 10.0

    def loss_fn(self, y_pred, y_true, dropout=True):
        batch_size = y_pred[0].shape[0]

        use_box_in_env_loss = self.collision_loss_weight != 0 or \
            self.offroad_loss_weight or \
            self.onrouting_loss_weight

        pred_points = y_pred[0].view(
            batch_size, -1) if self.pos_reg_loss_weight != 0 else None
        pred_boxs = y_pred[1].view(
            batch_size, -1) if self.box_loss_weight != 0 or \
            use_box_in_env_loss else None
        pred_pos_dists = y_pred[2].view(
            batch_size, -1)if self.pos_dist_loss_weight != 0 else None

        true_points = y_true[0].view(
            batch_size, -1) if self.pos_reg_loss_weight != 0 else None
        true_boxs = y_true[1].view(
            batch_size, -1) if self.box_loss_weight != 0 or \
            use_box_in_env_loss else None
        true_pos_dists = y_true[2].view(
            batch_size, -1) if self.pos_dist_loss_weight != 0 else None
        true_pred_obs = y_true[3].view(
            batch_size, -1) if self.collision_loss_weight != 0 else None
        true_offroad_mask = y_true[4].view(
            batch_size, -1) if self.offroad_loss_weight != 0 else None
        true_onrouting_mask = y_true[5].view(
            batch_size, -1) if self.onrouting_loss_weight != 0 else None
        is_synthesized = y_true[6].view(batch_size, -1)
        is_synthesized_weight = torch.ones(
            [batch_size, 1], device=pred_points.device)
        is_synthesized_weight[is_synthesized] = self.is_synthesized_weight
        is_turning = y_true[7].view(batch_size, -1)
        is_turning_weight = torch.ones(
            [batch_size, 1], device=pred_points.device)
        is_turning_weight[is_turning] = self.is_turning_weight

        weighted_pos_reg_loss = is_turning_weight * is_synthesized_weight \
            * self.pos_reg_loss_weight * torch.mean(nn.MSELoss(reduction='none')(
                pred_points, true_points), dim=1, keepdim=True) \
            if self.pos_reg_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        weighted_box_loss = is_turning_weight * is_synthesized_weight \
            * self.box_loss_weight * torch.mean(nn.BCELoss(reduction='none')(
                pred_boxs, true_boxs), dim=1, keepdim=True) \
            if self.box_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        weighted_pos_dist_loss = is_turning_weight * is_synthesized_weight \
            * self.pos_dist_loss_weight * torch.mean(nn.BCELoss(reduction='none')(
                pred_pos_dists, true_pos_dists), dim=1, keepdim=True) \
            if self.pos_dist_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        weighted_collision_loss = self.collision_loss_weight \
            * torch.mean(
                pred_boxs * true_pred_obs, dim=1, keepdim=True) \
            if self.collision_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        weighted_offroad_loss = self.offroad_loss_weight \
            * torch.mean(pred_boxs * true_offroad_mask, dim=1, keepdim=True) \
            if self.offroad_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        weighted_onrouting_loss = self.onrouting_loss_weight \
            * torch.mean(pred_boxs * true_onrouting_mask, dim=1, keepdim=True) \
            if self.onrouting_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        if self.losswise_focal_loss or self.batchwise_focal_loss:
            logging.info("original_weighted_pos_reg_loss is {}, "
                         "original_weighted_box_loss is {},"
                         "original_weighted_pos_dist_loss is {}, "
                         "original_weighted_collision_loss is {}, "
                         "original_weighted_offroad_loss is {}, "
                         "original_weighted_onrouting_loss is {}".
                         format(
                             torch.mean(weighted_pos_reg_loss),
                             torch.mean(weighted_box_loss),
                             torch.mean(weighted_pos_dist_loss),
                             torch.mean(weighted_collision_loss),
                             torch.mean(weighted_offroad_loss),
                             torch.mean(weighted_onrouting_loss)))

        if self.losswise_focal_loss:
            total_loss = torch.zeros((batch_size, 1), device=y_pred[0].device)\
                + self.kepsilon

            total_loss += clamped_exp(weighted_pos_reg_loss) \
                if self.pos_reg_loss_weight != 0 else 0

            total_loss += clamped_exp(weighted_box_loss) \
                if self.box_loss_weight != 0 else 0

            total_loss += clamped_exp(weighted_pos_dist_loss) \
                if self.pos_dist_loss_weight != 0 else 0

            total_loss += clamped_exp(weighted_collision_loss) \
                if self.collision_loss_weight != 0 else 0

            total_loss += clamped_exp(weighted_offroad_loss) \
                if self.offroad_loss_weight != 0 else 0

            total_loss += clamped_exp(weighted_onrouting_loss) \
                if self.onrouting_loss_weight != 0 else 0

            weighted_pos_reg_loss = (1 - clamped_exp(
                weighted_pos_reg_loss) / total_loss + self.kepsilon)**self.focal_loss_gamma \
                * weighted_pos_reg_loss \
                if self.pos_reg_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_box_loss = (1 - clamped_exp(
                weighted_box_loss) / total_loss + self.kepsilon)**self.focal_loss_gamma \
                * weighted_box_loss \
                if self.box_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_pos_dist_loss = (1 - clamped_exp(
                weighted_pos_dist_loss) / total_loss + self.kepsilon)**self.focal_loss_gamma \
                * weighted_pos_dist_loss \
                if self.pos_dist_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_collision_loss = (1 - clamped_exp(
                weighted_collision_loss) / total_loss + self.kepsilon)**self.focal_loss_gamma \
                * weighted_collision_loss \
                if self.collision_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_offroad_loss = (1 - clamped_exp(
                weighted_offroad_loss) / total_loss + self.kepsilon)**self.focal_loss_gamma \
                * weighted_offroad_loss \
                if self.offroad_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_onrouting_loss = (1 - clamped_exp(
                weighted_onrouting_loss) / total_loss + self.kepsilon)**self.focal_loss_gamma \
                * weighted_onrouting_loss \
                if self.onrouting_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        if self.batchwise_focal_loss:
            total_loss = torch.zeros((batch_size, 1), device=y_pred[0].device) \
                + self.kepsilon

            total_loss += weighted_pos_reg_loss \
                if self.pos_reg_loss_weight != 0 else 0

            total_loss += weighted_box_loss \
                if self.box_loss_weight != 0 else 0

            total_loss += weighted_pos_dist_loss \
                if self.pos_dist_loss_weight != 0 else 0

            total_loss += weighted_collision_loss \
                if self.collision_loss_weight != 0 else 0

            total_loss += weighted_offroad_loss \
                if self.offroad_loss_weight != 0 else 0

            total_loss += weighted_onrouting_loss \
                if self.onrouting_loss_weight != 0 else 0

            softmax_loss_weight = nn.Softmax(dim=0)(total_loss)

            weighted_pos_reg_loss = (1 - softmax_loss_weight + self.kepsilon) \
                ** self.focal_loss_gamma * weighted_pos_reg_loss \
                if self.pos_reg_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_box_loss = (1 - softmax_loss_weight + self.kepsilon) \
                ** self.focal_loss_gamma * weighted_box_loss \
                if self.box_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_pos_dist_loss = (1 - softmax_loss_weight + self.kepsilon) \
                ** self.focal_loss_gamma * weighted_pos_dist_loss \
                if self.pos_dist_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_collision_loss = (1 - softmax_loss_weight + self.kepsilon) \
                ** self.focal_loss_gamma * weighted_collision_loss \
                if self.collision_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_offroad_loss = (1 - softmax_loss_weight + self.kepsilon) \
                ** self.focal_loss_gamma * weighted_offroad_loss \
                if self.offroad_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

            weighted_onrouting_loss = (1 - softmax_loss_weight + self.kepsilon) \
                ** self.focal_loss_gamma * weighted_onrouting_loss \
                if self.onrouting_loss_weight != 0 else torch.zeros(1, device=y_pred[0].device)

        weighted_pos_reg_loss = torch.mean(weighted_pos_reg_loss)
        weighted_box_loss = torch.mean(weighted_box_loss)
        weighted_pos_dist_loss = torch.mean(weighted_pos_dist_loss)
        weighted_collision_loss = torch.mean(weighted_collision_loss)
        weighted_offroad_loss = torch.mean(weighted_offroad_loss)
        weighted_onrouting_loss = torch.mean(weighted_onrouting_loss)

        logging.info("weighted_pos_reg_loss is {}, weighted_box_loss is {},"
                     "weighted_pos_dist_loss is {}, weighted_collision_loss is {}, "
                     "weighted_offroad_loss is {}, weighted_onrouting_loss is {}".
                     format(
                         weighted_pos_reg_loss,
                         weighted_box_loss,
                         weighted_pos_dist_loss,
                         weighted_collision_loss,
                         weighted_offroad_loss,
                         weighted_onrouting_loss))

        imitation_dropout = False
        if dropout and self.imitation_dropout:
            imitation_dropout = self.imitation_dropout

        losses = dict()
        losses["weighted_pos_reg_loss"] = weighted_pos_reg_loss
        losses["weighted_box_loss"] = weighted_box_loss
        losses["weighted_pos_dist_loss"] = weighted_pos_dist_loss
        losses["weighted_collision_loss"] = weighted_collision_loss
        losses["weighted_offroad_loss"] = weighted_offroad_loss
        losses["weighted_onrouting_loss"] = weighted_onrouting_loss
        losses["total_loss"] = (0 if imitation_dropout and torch.rand(1) > 0.5 else 1) * \
            (weighted_pos_reg_loss
             + weighted_box_loss
             + weighted_pos_dist_loss) + \
            weighted_collision_loss + \
            weighted_offroad_loss + \
            weighted_onrouting_loss

        return losses

    def loss_info(self, y_pred, y_true):
        pred_points = y_pred[0]
        true_points = y_true[0]
        points_diff = pred_points - true_points
        # First two elements are assumed to be x and y position
        pose_diff = points_diff[:, :, 0:2]
        out = torch.sqrt(torch.sum(pose_diff ** 2, dim=-1))
        out = torch.mean(out)
        logging.info("Average displacement mse loss is {}".format(out))
        total_loss = self.loss_fn(y_pred, y_true, dropout=False)["total_loss"]
        return total_loss
