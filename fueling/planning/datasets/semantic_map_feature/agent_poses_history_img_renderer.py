#!/usr/bin/env python

import os
import shutil

import numpy as np
import cv2 as cv

from modules.planning.proto import learning_data_pb2


class AgentPosesHistoryImgRenderer(object):
    """class of AgentPosesHistoryImgRenderer to create a image of past ego car poses"""

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

    def _get_trans_point(self, p):
        point = np.round((p - self.local_base_point) / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def _get_affine_points(self, p):
        p = p - self.local_base_point
        theta = 3 * np.pi / 2 - self.local_base_heading
        point = np.dot(np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array(p).T).T
        point = np.round(point / self.resolution)
        return [self.local_base_point_w_idx + int(point[0]), self.local_base_point_h_idx - int(point[1])]

    def draw_agent_poses_history(self, ego_pose_history):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array(
            [ego_pose_history[0][0], ego_pose_history[0][1]])
        self.local_base_heading = ego_pose_history[0][2]
        for ego_pose in ego_pose_history:
            cv.circle(local_map, tuple(self._get_affine_points(
                np.array(ego_pose[:2]))), radius=4, color=(255))
        return local_map


if __name__ == "__main__":
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.55.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_agent_pose_history/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    agent_history_mapping = AgentPosesHistoryImgRenderer()
    for frame in offline_frames.learning_data:
        ego_pose_history = []
        for past_pose in frame.adc_trajectory_point:
            ego_pose_history.append(
                [past_pose.trajectory_point.path_point.x, past_pose.trajectory_point.path_point.y, past_pose.trajectory_point.path_point.theta])
        img = agent_history_mapping.draw_agent_poses_history(ego_pose_history)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir+"/ego_pos.npy"), ego_pos_dict)
