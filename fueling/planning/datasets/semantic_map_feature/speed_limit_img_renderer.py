#!/usr/bin/env python

import os
import shutil

import cv2 as cv
import numpy as np
import math

from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils


class SpeedLimitImgRenderer(object):
    """class of SpeedLimitImgRenderer to create a image of surrounding road speed limit"""

    def __init__(self, config_file, region):
        """contruct function to init RoadMapImgRenderer object"""
        self.map_dir = "/fuel/testdata/planning/semantic_map_features"
        self.base_map = cv.imread(os.path.join(self.map_dir, region + "_speedlimit.png"))
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = config.ego_idx_x
        self.local_base_point_h_idx = config.ego_idx_y  # lower center point in the image
        self.map_base_point_x = None
        self.map_base_point_y = None
        self.map_size_h = None
        self.map_size_w = None
        if region == "san_mateo":
            self.map_base_point_x = 558982.162591356
            self.map_base_point_y = 4156781.763329632
            self.map_size_h = 13785
            self.map_size_w = 10650
        elif region == "sunnyvale_with_two_offices":
            self.map_base_point_x = 585875.3316302994
            self.map_base_point_y = 4139916.6342316796
            self.map_size_h = 19311
            self.map_size_w = 27632
        elif region == "sunnyvale":
            self.map_base_point_x = 585875.3316302994
            self.map_base_point_y = 4139916.6342316796
            self.map_size_h = 19311
            self.map_size_w = 27632
        else:
            print("Chosen base map not created")
            exit()
        self.rough_crop_radius = int(
            math.sqrt(self.local_size_h**2 + self.local_size_w**2))

    def draw_speedlimit(self, center_x, center_y, center_heading, coordinate_heading=0.):
        center_point = np.array([center_x, center_y])
        center_basemap_idx = [int(np.round((center_point[0] - self.map_base_point_x) / self.resolution)),
                              int(self.map_size_h - np.round((center_point[1] - self.map_base_point_y) /
                                                             self.resolution))]
        rough_local_map = self.base_map[center_basemap_idx[1] - self.rough_crop_radius: center_basemap_idx[1] + self.rough_crop_radius,
                                        center_basemap_idx[0] - self.rough_crop_radius: center_basemap_idx[0] + self.rough_crop_radius]
        rough_local_map_grid = [
            2 * self.rough_crop_radius, 2 * self.rough_crop_radius]
        center_local_idx = [self.rough_crop_radius, self.rough_crop_radius]
        rotation_angle = 90 - np.degrees(center_heading) + np.degrees(coordinate_heading)
        M = cv.getRotationMatrix2D(
            tuple(center_local_idx), rotation_angle, 1.0)
        rotated = cv.warpAffine(
            rough_local_map, M, tuple(rough_local_map_grid))
        fine_crop = rotated[center_local_idx[1] - self.local_base_point_h_idx: center_local_idx[1] +
                            (self.local_size_h - self.local_base_point_h_idx), center_local_idx[0] -
                            self.local_base_point_w_idx: center_local_idx[0] + self.local_base_point_w_idx + 1]
        return fine_crop


if __name__ == '__main__':
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.55.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_local_speed_limit/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    speedlimit_mapping = SpeedLimitImgRenderer(config_file, "sunnyvale_with_two_offices")
    for frame in offline_frames.learning_data:
        img = speedlimit_mapping.draw_speedlimit(
            frame.localization.position.x, frame.localization.position.y, frame.localization.heading)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir + "/" + filename), img)
    np.save(os.path.join(output_dir + "/ego_pos.npy"), ego_pos_dict)
