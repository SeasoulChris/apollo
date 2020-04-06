#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2


class AgentPosesFutureImgRenderer(object):
    """class of AgentPosesFutureImgRenderer to create a image of past ego car poses"""

    def __init__(self):
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = int((self.local_size_w - 1) / 2)
        self.local_base_point_h_idx = 376  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.local_base_point = None
        self.local_base_heading = None
        self.max_future_time_horizon = 3  # second

        # TODO(Jinyun): read vehicle param from elsewhere
        self.front_edge_to_center = 3.89
        self.back_edge_to_center = 1.043
        self.left_edge_to_center = 1.055
        self.right_edge_to_center = 1.055

    def _get_trans_point(self, p):
        p = np.round(p / self.resolution)
        return [self.local_base_point_w_idx + int(p[0]), self.local_base_point_h_idx - int(p[1])]

    def _get_affine_points(self, p):
        p = p - self.local_base_point
        theta = np.pi / 2 - self.local_base_heading
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def _get_affine_ego_box(self, p, box_theta):
        p = p - self.local_base_point
        theta = np.pi / 2 - self.local_base_heading
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        box_theta_diff = np.pi / 2 + box_theta - self.local_base_heading
        corner_points = np.dot(np.array([[np.cos(box_theta_diff), -np.sin(box_theta_diff)],
                                         [np.sin(box_theta_diff), np.cos(box_theta_diff)]]),
                               np.array([[self.front_edge_to_center, self.front_edge_to_center,
                                          -self.back_edge_to_center, -self.back_edge_to_center],
                                         [self.left_edge_to_center, -self.right_edge_to_center,
                                          -self.right_edge_to_center, self.left_edge_to_center]])).T + point
        corner_points = [self._get_trans_point(
            point) for point in corner_points]
        return np.int32(corner_points)

    def draw_agent_future_trajectory(self, frame_time_sec, center_x, center_y, center_heading, ego_pose_future):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        current_time = frame_time_sec
        print("current_time is {}".format(current_time))
        for ego_pose in ego_pose_future:
            relative_time = ego_pose.timestamp_sec - current_time
            print("prediction_time is {}".format(ego_pose.timestamp_sec))
            if relative_time > self.max_future_time_horizon:
                break
            color = relative_time / self.max_future_time_horizon * 255
            traj_points = tuple(self._get_affine_points(
                np.array([ego_pose.trajectory_point.path_point.x, ego_pose.trajectory_point.path_point.y])))
            for traj_point in traj_points:
                if traj_point[0] < 0 or traj_point[0] > self.local_size_h or traj_point[1] < 0 or traj_point[1] > self.local_size_h:
                    print("draw_agent_future_trajectory out of canvas bound")
                    return local_map
            cv.circle(local_map, traj_points, radius=4, color=color)
        return local_map

    def draw_agent_pose_future(self, center_x, center_y, center_heading, ego_pose_future, timestamp_idx):
        '''
        It uses index to get specific frame in the future rather than timestamp. Make sure to inspect and clean data before using it
        '''
        if len(ego_pose_future) < timestamp_idx:
            print("timestamp_idx larger than what is available")
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        ego_pose = ego_pose_future[timestamp_idx]
        idx = self._get_affine_points(
            np.array([ego_pose.trajectory_point.path_point.x, ego_pose.trajectory_point.path_point.y]))
        if idx[0] < 0 or idx[0] > self.local_size_h or idx[1] < 0 or idx[1] > self.local_size_h:
            print("draw_agent_pose_future out of canvas bound")
            return local_map
        local_map[idx[1], idx[0]] = 255
        return local_map

    def draw_agent_box_future(self, center_x, center_y, center_heading, ego_pose_future, timestamp_idx):
        '''
        It uses index to get specific frame in the future rather than timestamp. Make sure to inspect and clean data before using it
        '''
        if len(ego_pose_future) < timestamp_idx:
            print("timestamp_idx larger than what is available")
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading
        ego_pose = ego_pose_future[timestamp_idx]
        corner_points = self._get_affine_ego_box(
            np.array([ego_pose.trajectory_point.path_point.x, ego_pose.trajectory_point.path_point.y]), ego_pose.trajectory_point.path_point.theta)
        for corner_point in corner_points:
            if corner_point[0] < 0 or corner_point[0] > self.local_size_h or corner_point[1] < 0 or corner_point[1] > self.local_size_h:
                print("draw_agent_box_future out of canvas bound")
                return local_map
        cv.fillPoly(local_map, [corner_points], color=255)
        return local_map


if __name__ == "__main__":
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/2019-10-17-13-36-41/learning_data.66.bin.future_status.bin",
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
    agent_future_mapping = AgentPosesFutureImgRenderer()
    for frame in offline_frames.learning_data:
        # img = agent_future_mapping.draw_agent_future_trajectory(frame.timestamp_sec,
        #                                                         frame.localization.position.x,
        #                                                         frame.localization.position.y,
        #                                                         frame.localization.heading,
        #                                                         frame.output.adc_future_trajectory_point)
        # key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        # filename = key + ".png"
        # ego_pos_dict[key] = [frame.localization.position.x,
        #                      frame.localization.position.y, frame.localization.heading]
        # cv.imwrite(os.path.join(output_dir, filename), img)
        for i in range(30):
            img = agent_future_mapping.draw_agent_box_future(frame.localization.position.x,
                                                             frame.localization.position.y,
                                                             frame.localization.heading,
                                                             frame.output.adc_future_trajectory_point,
                                                             i)
            key = "{}@{}@{:.3f}".format(
                frame.frame_num, i, frame.timestamp_sec)
            filename = key + ".png"
            cv.imwrite(os.path.join(output_dir, filename), img)

    np.save(os.path.join(output_dir+"/ego_pos.npy"), ego_pos_dict)
