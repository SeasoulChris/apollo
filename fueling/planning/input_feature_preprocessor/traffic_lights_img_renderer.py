#!/usr/bin/env python
import os
import shutil

import numpy as np
import cv2 as cv

from modules.map.proto import map_pb2
from modules.map.proto import map_lane_pb2
from modules.map.proto import map_signal_pb2
from modules.map.proto import map_overlap_pb2
from modules.perception.proto import traffic_light_detection_pb2
from modules.planning.proto import learning_data_pb2
from modules.planning.proto import planning_semantic_map_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.planning.input_feature_preprocessor.renderer_utils as renderer_utils


class TrafficLightsImgRenderer(object):
    """class of TrafficLightsImgRenderer to create images of surrounding traffic conditions"""

    def __init__(self, config_file, region_base_map_data_dict):
        config = proto_utils.get_pb_from_text_file(config_file,
                                                   planning_semantic_map_config_pb2.
                                                   PlanningSemanticMapConfig())
        self.region_base_map_data_dict = region_base_map_data_dict
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y])  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.regions_lane_dict = {}
        self.regions_signal_dict = {}
        self.regions_overlap_dict = {}
        self._load_lane()
        self._load_traffic_light()
        self._load_overlap()

    def _load_lane(self):
        for region, hdmap in self.region_base_map_data_dict.items():
            region_lane_dict = {}
            for lane in hdmap.lane:
                region_lane_dict[lane.id.id] = lane
            self.regions_lane_dict[region] = region_lane_dict

    def _load_traffic_light(self):
        for region, hdmap in self.region_base_map_data_dict.items():
            region_signal_dict = {}
            for signal in hdmap.signal:
                region_signal_dict[signal.id.id] = signal
            self.regions_signal_dict[region] = region_signal_dict

    def _load_overlap(self):
        for region, hdmap in self.region_base_map_data_dict.items():
            region_overlap_dict = {}
            for overlap in hdmap.overlap:
                region_overlap_dict[overlap.id.id] = overlap
            self.regions_overlap_dict[region] = region_overlap_dict

    def _get_traffic_light_by_id(self, region, id):
        return self.regions_signal_dict[region][id]

    def _get_lane_by_id(self, region, id):
        return self.regions_lane_dict[region][id]

    def _get_overlap_by_id(self, region, id):
        return self.regions_overlap_dict[region][id]

    def draw_traffic_lights(self, region, center_x, center_y, center_heading,
                            observed_traffic_lights, coordinate_heading=0.):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        local_base_point = np.array([center_x, center_y])
        local_base_heading = center_heading

        for traffic_light_status in observed_traffic_lights:
            traffic_light = self._get_traffic_light_by_id(
                region, traffic_light_status.id)
            traffic_light_color = (255)
            if traffic_light_status.color == traffic_light_detection_pb2.TrafficLight.RED:
                traffic_light_color = (255)
            elif traffic_light_status.color == traffic_light_detection_pb2.TrafficLight.YELLOW:
                traffic_light_color = (192)
            else:
                traffic_light_color = (128)

            for overlap_id in traffic_light.overlap_id:
                overlap = self._get_overlap_by_id(region, overlap_id.id)
                for overlap_object in overlap.object:
                    if overlap_object.HasField("lane_overlap_info"):
                        lane = self._get_lane_by_id(
                            region, overlap_object.id.id)
                        for segment in lane.central_curve.segment:
                            for i in range(len(segment.line_segment.point) - 1):
                                p0 = tuple(renderer_utils.get_img_idx(
                                    renderer_utils.point_affine_transformation(
                                        np.array([segment.line_segment.point[i].x,
                                                  segment.line_segment.point[i].y]),
                                        local_base_point,
                                        np.pi / 2 - local_base_heading + coordinate_heading),
                                    self.local_base_point_idx,
                                    self.resolution))
                                p1 = tuple(renderer_utils.get_img_idx(
                                    renderer_utils.point_affine_transformation(
                                        np.array([segment.line_segment.point[i + 1].x,
                                                  segment.line_segment.point[i + 1].y]),
                                        local_base_point,
                                        np.pi / 2 - local_base_heading + coordinate_heading),
                                    self.local_base_point_idx,
                                    self.resolution))
                                cv.line(local_map, tuple(p0), tuple(p1),
                                        color=traffic_light_color, thickness=4)
        return local_map
