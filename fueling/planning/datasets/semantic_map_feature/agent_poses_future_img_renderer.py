#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils


class AgentPosesFutureImgRenderer(object):
    """class of AgentPosesFutureImgRenderer to create a image of past ego car poses"""

    def __init__(self, config_file):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y])  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.local_base_point = None
        self.local_base_heading = None
        self.max_future_time_horizon = config.max_ego_future_horizon  # second

        # TODO(Jinyun): read vehicle param from elsewhere
        self.front_edge_to_center = 3.89
        self.back_edge_to_center = 1.043
        self.left_edge_to_center = 1.055
        self.right_edge_to_center = 1.055
        self.east_oriented_box = np.array([[self.front_edge_to_center, self.front_edge_to_center,
                                            -self.back_edge_to_center, -self.back_edge_to_center],
                                           [self.left_edge_to_center, -self.right_edge_to_center,
                                            -self.right_edge_to_center, self.left_edge_to_center]]).T

    def draw_agent_future_trajectory(self, frame_time_sec, center_x,
                                     center_y, center_heading, ego_pose_future):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        current_time = frame_time_sec
        # print("current_time is {}".format(current_time))
        for ego_pose in ego_pose_future:
            relative_time = ego_pose.timestamp_sec - current_time
            # print("prediction_time is {}".format(ego_pose.timestamp_sec))
            if relative_time > self.max_future_time_horizon:
                break
            color = relative_time / self.max_future_time_horizon * 255
            traj_point = tuple(renderer_utils.get_img_idx(
                renderer_utils.point_affine_transformation(
                    np.array([ego_pose.trajectory_point.path_point.x,
                              ego_pose.trajectory_point.path_point.y]),
                    self.local_base_point,
                    np.pi / 2 - self.local_base_heading),
                self.local_base_point_idx,
                self.resolution))
            if traj_point[0] < 0 or traj_point[0] > self.local_size_h or traj_point[1] < 0 or traj_point[1] > self.local_size_w:
                # print("draw_agent_future_trajectory out of canvas bound")
                return local_map
            cv.circle(local_map, traj_point, radius=4, color=color)
        return local_map

    def draw_agent_pose_future(self, center_x, center_y, center_heading,
                               ego_pose_future, timestamp_idx):
        '''
        It uses index to get specific frame in the future rather than timestamp. Make sure to inspect and clean data before using it
        '''
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        if len(ego_pose_future) <= timestamp_idx:
            # print("timestamp_idx larger than what is available in agent pose future")
            return local_map
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        ego_pose = ego_pose_future[timestamp_idx]
        idx = tuple(renderer_utils.get_img_idx(
            renderer_utils.point_affine_transformation(
                np.array([ego_pose.trajectory_point.path_point.x,
                          ego_pose.trajectory_point.path_point.y]),
                self.local_base_point,
                np.pi / 2 - self.local_base_heading),
            self.local_base_point_idx,
            self.resolution))
        if idx[0] < 0 or idx[0] > self.local_size_h or idx[1] < 0 or idx[1] > self.local_size_w:
            # print("draw_agent_pose_future out of canvas bound")
            return local_map
        local_map[idx[1], idx[0]] = 255
        return local_map

    def draw_agent_box_future(self, center_x, center_y, center_heading,
                              ego_pose_future, timestamp_idx):
        '''
        It uses index to get specific frame in the future rather than timestamp. Make sure to inspect and clean data before using it
        '''
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        if len(ego_pose_future) <= timestamp_idx:
            # print("timestamp_idx larger than what is availablein agent box future")
            return local_map
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        ego_pose = ego_pose_future[timestamp_idx]
        ego_path_point = np.array([ego_pose.trajectory_point.path_point.x, ego_pose.trajectory_point.path_point.y])
        box_theta = ego_pose.trajectory_point.path_point.theta + np.pi / 2 - self.local_base_heading
        theta = np.pi / 2 - self.local_base_heading
        corner_points = renderer_utils.box_affine_tranformation(self.east_oriented_box,
                                                                ego_path_point,
                                                                box_theta,
                                                                self.local_base_point, 
                                                                theta,
                                                                self.local_base_point_idx,
                                                                self.resolution)
        for corner_point in corner_points:
            if corner_point[0] < 0 or corner_point[0] > self.local_size_h or corner_point[1] < 0 or corner_point[1] > self.local_size_w:
                # print("draw_agent_box_future out of canvas bound")
                return local_map
        cv.fillPoly(local_map, [corner_points], color=255)
        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/test_interpolated_data/00001.record.0.bin.future_status.bin",
              'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_agent_pose_future/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    agent_future_mapping = AgentPosesFutureImgRenderer(config_file)

    for frame in offline_frames.learning_data:
        img = agent_future_mapping.draw_agent_future_trajectory(frame.timestamp_sec,
                                                                frame.localization.position.x,
                                                                frame.localization.position.y,
                                                                frame.localization.heading,
                                                                frame.output.adc_future_trajectory_point)
        # img = agent_future_mapping.draw_agent_box_future(frame.localization.position.x,
        #                                                  frame.localization.position.y,
        #                                                  frame.localization.heading,
        #                                                  frame.output.adc_future_trajectory_point, 10)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)

    np.save(os.path.join(output_dir + "/ego_pos.npy"), ego_pos_dict)
