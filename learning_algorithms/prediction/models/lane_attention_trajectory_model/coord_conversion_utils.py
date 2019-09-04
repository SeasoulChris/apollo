#!/usr/bin/env python

import torch
import torch.nn as nn

from learning_algorithms.utilities.train_utils import *
from learning_algorithms.utilities.network_utils import *


class PointToLineProjection(nn.Module):
    '''Get the projection point from a given point to a given line-segment.
    '''

    def __init__(self):
        super(PointToLineProjection, self).__init__()

    def forward(self, line_seg_start, line_seg_end, point):
        '''
        params:
            - line_seg_start: (N x 2)
            - line_seg_end: (N x 2)
            - point: (N x 2)

        return:
            - projected_point: (N x 2)
            - dist: (N x 1)
        '''
        N = point.size(0)
        # (N x 2)
        line_vec = line_seg_end - line_seg_start
        # (N x 1)
        line_vec_mag = torch.sqrt(torch.sum(line_vec**2, 1)).view(N, 1)
        # (N x 2)
        unit_line_vec = line_vec / line_vec_mag.repeat(1, 2)

        # (N x 2)
        point_vec = point - line_seg_start
        # (N x 1)
        projected_point_vec_mag = torch.sum(point_vec * unit_line_vec, 1).view(N, 1)
        # (N x 2)
        projected_point_vec = projected_point_vec_mag.repeat(1, 2) * unit_line_vec
        # (N x 2)
        projected_point = line_seg_start + projected_point_vec
        # (N x 1)
        dist = torch.sqrt(torch.sum((point - projected_point)**2, 1)).view(N, 1)

        return projected_point, dist


class FindClosestNodeFromLineToPoint(nn.Module):
    '''Given a line(consisting of multiple nodes), find the node with the
       shortest distance to a given point.
    '''

    def __init__(self):
        super(FindClosestNodeFromLineToPoint, self).__init__()

    def forward(self, line_nodes, point):
        '''
        params:
            - line_nodes: N x num_node x 2
            - point: N x 2

        return:
            - idx of the min dist node: N
        '''
        N = point.size(0)
        num_node = line_nodes.size(1)
        # (N x num_node-2 x 2)
        nodes = line_nodes[:, 1:-1, :2].float()

        # Calculate the L2 distance between every lane-points to the point.
        # (N x num_node-2 x 2)
        distances = nodes - point.view(N, 1, 2).repeat(1, num_node-2, 1).float()
        distances = distances ** 2
        # (N x num_node-2)
        distances = torch.sum(distances, 2)

        # Figure out the idx of the lane-point that's closest to obstacle.
        # (N)
        min_idx = torch.argmin(distances, dim=1)

        return min_idx + 1


class FindClosestLineSegmentFromLineToPoint(nn.Module):
    def __init__(self):
        super(FindClosestLineSegmentFromLineToPoint, self).__init__()
        self.find_closest_idx = FindClosestNodeFromLineToPoint()

    def forward(self, line_nodes, point):
        '''
        params:
            - line_nodes: N x num_node x 2
            - point: N x 2

        return:
            - start idx of the min dist node: N
            - end idx of the min dist node: N
        '''
        N = line_nodes.size(0)
        # (N)
        min_idx = self.find_closest_idx(line_nodes, point)
        # (N x 2)
        min_node = line_nodes[torch.arange(N), min_idx, :]

        # (N)
        min_idx_prev = min_idx - 1
        # (N x 2)
        min_node_prev = line_nodes[torch.arange(N), min_idx_prev, :]
        # (N)
        dist_to_prev = torch.sum((min_node_prev - min_node) ** 2, 1)

        # (N)
        min_idx_next = min_idx + 1
        # (N x 2)
        min_node_next = line_nodes[torch.arange(N), min_idx_next, :]
        # (N )
        dist_to_next = torch.sum((min_node_next - min_node) ** 2, 1)

        # Get the 2nd minimum distance node's index.
        min_idx_2nd = min_idx_prev
        idx_to_be_modified = (dist_to_next < dist_to_prev)
        if torch.sum(idx_to_be_modified).long().item() != 0:
            min_idx_2nd[idx_to_be_modified] = min_idx_next[idx_to_be_modified]

        # Return the correct order
        min_indices = torch.cat((min_idx.view(N, 1), min_idx_2nd.view(N, 1)), 1)
        idx_before, _ = torch.min(min_indices, dim=1)
        idx_after, _ = torch.max(min_indices, dim=1)
        return idx_before, idx_after


class ProjPtToSL(nn.Module):
    def __init__(self):
        super(ProjPtToSL, self).__init__()

    def forward(self, proj_pt, dist, idx_before, idx_after, lane_features):
        '''
        params:
            - proj_pt: N x 2
            - dist: N x 2
            - idx_before: N
            - idx_after: N
            - lane_features: N x 150 x 4
        '''
        N = lane_features.size(0)
        num_lane_pt = lane_features.size(1)

        # Get the distance of each lane-pt w.r.t. the 0th one.
        # (N x 150)
        lane_pt_spacing = cuda(torch.zeros(N, num_lane_pt))
        lane_pt_spacing[:, 1:] = torch.sqrt(torch.sum(
            (lane_features[:, 1:, :2] - lane_features[:, :-1, :2]) ** 2, 2))
        # (N x 150) The distance of each lane-pt w.r.t. the 0th one.
        lane_pt_dist = torch.cumsum(lane_pt_spacing, 1)

        # Get the distance of the proj_pt to the pt of idx_before.
        # (N x 2)
        pt_before = lane_features[torch.arange(N), idx_before, :2].float()
        pt_after = lane_features[torch.arange(N), idx_after, :2].float()
        # (N x 2)
        line_seg_vec = pt_after - pt_before
        # (N x 1)
        line_seg_vec_mag = torch.sqrt(torch.sum(line_seg_vec**2, 1)).view(N, 1)
        # (N x 2)
        line_seg_vec_unit = line_seg_vec / line_seg_vec_mag.repeat(1, 2)
        # (N)
        dist_to_pt_before = torch.sum((proj_pt - pt_before) * line_seg_vec_unit, 1)

        # Get the S-coord.
        # (N x 1)
        S = (lane_pt_dist[torch.arange(N), idx_before] + dist_to_pt_before).view(N, 1)

        # Get the L-coord.
        # (N x 1)
        L = (dist[:, 0] * line_seg_vec_unit[:, 1] - dist[:, 1] * line_seg_vec_unit[:, 0]).view(N, 1)

        # (N x 2)
        SL = torch.cat((S, L), 1)
        return SL


class SLToXY(nn.Module):
    def __init__(self):
        super(SLToXY, self).__init__()

    def forward(self, lane_features, pt_sl):
        '''
        params:
            - lane_features: N x 150 x 4
            - pt_sl: N x 2 (for now, assume l=0, TODO: eliminate this assumption)
        return:
            - XY: N x 2
        '''
        N = lane_features.size(0)
        num_lane_pt = lane_features.size(1)
        # Get the distance of each lane-pt w.r.t. the 0th one.
        # (N x 150)
        lane_pt_spacing = cuda(torch.zeros(N, num_lane_pt))
        lane_pt_spacing[:, 1:] = torch.sqrt(torch.sum(
            (lane_features[:, 1:, :2] - lane_features[:, :-1, :2]) ** 2, 2))
        lane_pt_dist = torch.cumsum(lane_pt_spacing, 1)

        # Get the idx_before and idx_after
        # (N)
        mask_front = (pt_sl[:, 0] < lane_pt_dist[:, 1])
        mask_back = (pt_sl[:, 0] > lane_pt_dist[:, -2])
        mask_middle = (mask_front == 0) * (mask_back == 0)
        idx_before = cuda(torch.zeros(N))
        if torch.sum(mask_back).long() != 0:
            idx_before[mask_back] = ((num_lane_pt - 2) * cuda(torch.ones(N)))[mask_back]
        S = pt_sl[:, 0].view(N, 1)
        S_repeated = S.repeat(1, num_lane_pt)
        # (N x 150)
        s_mask = S_repeated < lane_pt_dist
        # (N x 149)
        s_mask = s_mask[:, 1:].long() - s_mask[:, :-1].long()
        # (N)
        s_mask = torch.argmax(s_mask, dim=1)
        idx_before[mask_middle] = s_mask[mask_middle].float()
        idx_before = idx_before.long()
        idx_after = idx_before + 1

        # Get the pt_before and pt_after.
        # (N x 2)
        pt_before = lane_features[torch.arange(N), idx_before, :2]
        pt_after = lane_features[torch.arange(N), idx_after, :2]

        # Get the actual s w.r.t. each line-segment of interest.
        # (N)
        s_actual = S - lane_pt_dist[torch.arange(N), idx_before].view(N, 1)

        # Get the unit vector of (pt_before, pt_after)
        # (N x 2)
        line_seg_vec = pt_after - pt_before
        # (N x 1)
        line_seg_vec_mag = torch.sqrt(torch.sum(line_seg_vec**2, 1)).view(N, 1)
        # (N x 2)
        line_seg_vec_unit = line_seg_vec / line_seg_vec_mag.repeat(1, 2)

        # Get the actual XY point
        XY = pt_before + s_actual * line_seg_vec_unit

        return XY


class BroadcastObstaclesToLanes(nn.Module):
    '''There are N obstacles and M corresponding lanes (N <= M).
       We need to broadcast the number of obstacles to be M.
    '''

    def __init__(self):
        super(BroadcastObstaclesToLanes, self).__init__()

    def forward(self, obs_pos, same_obs_mask):
        '''
        params:
            - obs_pos: N x 2
            - same_obs_mask: M x 1
        return:
            - repeated_obs_pos: M x 2
        '''
        M = same_obs_mask.size(0)
        # (M x 2)
        repeated_obs_pos = cuda(torch.zeros(M, 2))

        for obs_id in range(same_obs_mask.max().long().item() + 1):
            curr_mask = (same_obs_mask[:, 0] == obs_id)
            curr_num_lane = torch.sum(curr_mask).long().item()
            # (curr_num_lane x 2)
            curr_obs_pos = obs_pos[obs_id, :].view(1, 2)
            curr_obs_pos = curr_obs_pos.repeat(curr_num_lane, 1)
            repeated_obs_pos[curr_mask, :] = curr_obs_pos.float()

        return repeated_obs_pos


class ObstacleToLaneRelation(nn.Module):
    '''Calculate the distance of an obstacle to a certain lane.
    '''

    def __init__(self):
        super(ObstacleToLaneRelation, self).__init__()
        self.broadcasting = BroadcastObstaclesToLanes()
        self.get_projection_point = PointToLineProjection()
        self.find_the_closest_two_points = FindClosestLineSegmentFromLineToPoint()

    def forward(self, lane_features, obs_pos, same_obs_mask):
        '''
        params:
            - lane_features: M x 150 x 4
            - obs_pos: N x 2
            - same_obs_mask: M x 1
        return:
            - projected_points: M x 2
            - idx_before and idx_after: M x 2
        '''
        N = obs_pos.size(0)
        M = lane_features.size(0)
        lane_features = lane_features.float()
        # (M x 2)
        repeated_obs_pos = self.broadcasting(obs_pos, same_obs_mask)
        # (M)
        idx_before, idx_after = self.find_the_closest_two_points(lane_features, repeated_obs_pos)
        indices = torch.cat((idx_before.view(M, 1), idx_after.view(M, 1)), 1)
        # (M x 2)
        proj_pt, _ = self.get_projection_point(
            lane_features[torch.arange(M), idx_before, :2], lane_features[torch.arange(M), idx_after, :2], repeated_obs_pos)

        return proj_pt, indices, repeated_obs_pos
