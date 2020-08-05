#!/usr/bin/env python
import math

import cv2 as cv
import numpy as np

from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.input_feature_preprocessor.renderer_utils as renderer_utils


class OffroadMaskImgRenderer(object):
    """class of OffroadMaskImgRenderer to create an road map around ego vehicle with map element """

    def __init__(self, config_file, region_base_offroad_img_dict):
        """contruct function to init OffroadMaskImgRenderer object"""
        config = proto_utils.get_pb_from_text_file(config_file,
                                                   planning_semantic_map_config_pb2.
                                                   PlanningSemanticMapConfig())
        self.region_base_offroad_img_dict = region_base_offroad_img_dict
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        # lower center point in the image
        self.local_base_point_w_idx = config.ego_idx_x
        self.local_base_point_h_idx = config.ego_idx_y  # lower center point in the image
        self.rough_crop_radius = int(
            math.sqrt(self.local_size_h**2 + self.local_size_w**2))

    def draw_offroad_mask(self, region, center_x, center_y, center_heading, coordinate_heading=0.):
        base_offroad_mask = self.region_base_offroad_img_dict[region]
        map_size_h = base_offroad_mask.shape[0]
        map_base_point_x = None
        map_base_point_y = None
        if region == "san_mateo":
            map_base_point_x = 558982.162591356
            map_base_point_y = 4156781.763329632
        elif region == "sunnyvale_with_two_offices":
            map_base_point_x = 585875.3316302994
            map_base_point_y = 4139916.6342316796
        elif region == "sunnyvale":
            map_base_point_x = 585875.3316302994
            map_base_point_y = 4139916.6342316796
        else:
            print("Chosen base map not created")
            exit()
        map_base_point = np.array(
            [map_base_point_x, map_base_point_y])
        map_base_point_idx = np.array([0, map_size_h])

        center_point = np.array([center_x, center_y])
        center_basemap_idx = renderer_utils.get_img_idx(
            center_point - map_base_point, map_base_point_idx, self.resolution)
        rough_local_map = base_offroad_mask[center_basemap_idx[1] - self.rough_crop_radius:
                                            center_basemap_idx[1]
                                            + self.rough_crop_radius,
                                            center_basemap_idx[0] - self.rough_crop_radius:
                                            center_basemap_idx[0] + self.rough_crop_radius]
        rough_local_map_grid = [
            2 * self.rough_crop_radius, 2 * self.rough_crop_radius]
        center_local_idx = [self.rough_crop_radius, self.rough_crop_radius]
        rotation_angle = 90 - \
            np.degrees(center_heading) + np.degrees(coordinate_heading)
        M = cv.getRotationMatrix2D(
            tuple(center_local_idx), rotation_angle, 1.0)
        rotated = cv.warpAffine(
            rough_local_map, M, tuple(rough_local_map_grid))
        fine_crop = rotated[center_local_idx[1] - self.local_base_point_h_idx:
                            center_local_idx[1]
                            + (self.local_size_h
                               - self.local_base_point_h_idx), center_local_idx[0]
                            - self.local_base_point_w_idx:
                            center_local_idx[0] + self.local_base_point_w_idx]
        return fine_crop
