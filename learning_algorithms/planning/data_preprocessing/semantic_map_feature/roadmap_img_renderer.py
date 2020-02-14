#!/usr/bin/env python

import os
import shutil

import cv2 as cv
import numpy as np

from modules.prediction.proto import offline_features_pb2
from modules.prediction.proto import semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils


class RoadMapImgRenderer(object):
    """class of RoadMapImgRenderer to create an road map around ego vehicle with map element """

    def __init__(self, region):
        """contruct function to init RoadMapImgRenderer object"""
        self.map_dir = "/apollo/modules/data/fuel/learning_algorithms/planning/data_preprocessing/"
        self.base_map = cv.imread(os.path.join(map_dir, region + ".png"))
        # TODO(Jinyun): add config proto and modify according to chauffeurnet
        config = semantic_map_config_pb2.SemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(
            os.path.join(map_dir, "semantic_map_config.pb.txt"), config)
        self.resolution = config.resolution
        self.observation_range = config.observation_range
        self.map_base_point_x = config.base_point.x
        self.map_base_point_y = config.base_point.y
        self.local_base_point =
        self.dim_y = config.dim_y

    def get_trans_point(self, p, local_base_point, GRID):
        point = np.round((p - local_base_point) / self.resolution)
        return [int(point[0]), GRID[1] - int(point[1])]

    def draw_roadmap(self, center_x, center_y, center_heading):
        # TODO(Jinyun): modify drawing according to chauffeurnet
        center_point = np.array([center_x, center_y])
        center_idx = [int(np.round((center_point[0]-self.map_base_point_x)/self.resolution)),
                      int(self.dim_y-np.round((center_point[1]-self.map_base_point_y) /
                                              self.resolution))]
        local_base_point = np.array(center_point) - self.observation_range
        GRID = [int(2 * self.observation_range / self.resolution),
                int(2 * self.observation_range / self.resolution)]
        center = tuple(self.get_trans_point(center_point, local_base_point, GRID))
        feature_map = self.base_map[center_idx[1]-int(self.observation_range /
                                                      self.resolution):center_idx[1]+int(self.observation_range /
                                                                                         self.resolution),
                                    center_idx[0]-int(self.observation_range /
                                                      self.resolution):center_idx[0]+int(self.observation_range /
                                                                                         self.resolution)]
        heading_angle = center_heading * 180 / np.pi
        M = cv.getRotationMatrix2D(center, 90-heading_angle, 1.0)
        rotated = cv.warpAffine(feature_map, M, tuple(GRID))
        output = rotated[center[1]-300:center[1] +
                         100, center[0]-200:center[0]+200]
        return cv.resize(output, (224, 224))


if __name__ == '__main__':
    # TODO(All): build offline_features for chauffeurnet
    list_frame = offline_features_pb2.ListFrameEnv()
    with open("/apollo/data/frame_env.0.bin", 'r') as file_in:
        list_frame.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    roadmap_mapping = RoadMapImgRenderer("san_mateo")

    # TODO(Jinyun): drawing the corresponding roadmap for each frame

    np.save(os.path.join(output_dir+"/ego_pos.npy"), ego_pos_dict)
