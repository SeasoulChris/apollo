#!/usr/bin/env python

import os
import shutil

import cv2 as cv
import numpy as np
import math

from modules.planning.proto import learning_data_pb2


class RoadMapImgRenderer(object):
    """class of RoadMapImgRenderer to create an road map around ego vehicle with map element """

    def __init__(self, region):
        """contruct function to init RoadMapImgRenderer object"""
        self.map_dir = "/fuel/learning_algorithms/planning/data_preprocessing/semantic_map_feature"
        self.base_map = cv.imread(os.path.join(self.map_dir, region + ".png"))
        # TODO(Jinyun): use config file
        self.resolution = 0.1  # in meter/pixel
        self.local_size_h = 501  # H * W image
        self.local_size_w = 501  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = int((self.local_size_w - 1) / 2)
        self.local_base_point_h_idx = 376  # lower center point in the image
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
            self.map_size_h = 27632
            self.map_size_w = 19311
        else:
            print("Chosen base map not created")
            exit()

        self.rough_crop_radius = int(
            math.sqrt(self.local_size_h**2 + self.local_size_w**2))
        self.rough_crop_radius = 2000

    def get_trans_point(self, p, local_base_point, GRID):
        point = np.round((p - local_base_point) / self.resolution)
        return [int(point[0]), GRID[1] - int(point[1])]

    def draw_roadmap(self, center_x, center_y, center_heading):
        center_point = np.array([center_x, center_y])
        center_basemap_idx = [int(np.round((center_point[0] - self.map_base_point_x) / self.resolution)),
                              int(self.map_size_h - np.round((center_point[1]-self.map_base_point_y) /
                                                             self.resolution))]
        rough_local_map = self.base_map[(center_basemap_idx[1] - self.rough_crop_radius): (center_basemap_idx[1] + self.rough_crop_radius),
                                        (center_basemap_idx[0] - self.rough_crop_radius): (center_basemap_idx[0] + self.rough_crop_radius)]
        # return rough_local_map
        print(rough_local_map.shape)
        rough_local_map_grid = [
            2 * self.rough_crop_radius, 2 * self.rough_crop_radius]
        center_local_idx = [self.rough_crop_radius, self.rough_crop_radius]
        rotation_angle = 90 - center_heading * 180 / np.pi
        M = cv.getRotationMatrix2D(
            tuple(center_local_idx), rotation_angle, 1.0)
        rotated = cv.warpAffine(
            rough_local_map, M, tuple(rough_local_map_grid))
        fine_crop = rotated[center_local_idx[1] - self.local_base_point_h_idx: center_local_idx[1] +
                            (self.local_size_h - self.local_base_point_h_idx), center_local_idx[0] -
                            self.local_base_point_w_idx: center_local_idx[0] + self.local_base_point_w_idx]
        return fine_crop


if __name__ == '__main__':
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.66.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    roadmap_mapping = RoadMapImgRenderer("sunnyvale_with_two_offices")
    for frame in offline_frames.learning_data:
        img = roadmap_mapping.draw_roadmap(
            frame.localization.position.x, frame.localization.position.y, frame.localization.heading)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir + "/" + filename), img)
    np.save(os.path.join(output_dir+"/ego_pos.npy"), ego_pos_dict)
