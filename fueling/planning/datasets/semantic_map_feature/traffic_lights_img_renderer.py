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
import fueling.planning.datasets.semantic_map_feature.renderer_utils as renderer_utils

class TrafficLightsImgRenderer(object):
    """class of TrafficLightsImgRenderer to create images of surrounding traffic conditions"""

    def __init__(self, config_file, region):
        config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        config = proto_utils.get_pb_from_text_file(config_file, config)
        self.resolution = config.resolution  # in meter/pixel
        self.local_size_h = config.height  # H * W image
        self.local_size_w = config.width  # H * W image
        self.local_base_point_idx = np.array(
            [config.ego_idx_x, config.ego_idx_y])  # lower center point in the image
        self.GRID = [self.local_size_w, self.local_size_h]
        self.center = None
        self.center_heading = None

        self.region = region
        self.hd_map = None
        self.signal_dict = {}
        self.lane_dict = {}
        self.overlap_dict = {}
        self._read_hdmap()
        self._load_traffic_light()
        self._load_lane()
        self._load_overlap()

    def _read_hdmap(self):
        """read the hdmap from base_map.bin"""
        self.hd_map = map_pb2.Map()
        map_path = "/apollo/modules/map/data/" + self.region + "/base_map.bin"
        try:
            with open(map_path, 'rb') as file_in:
                self.hd_map.ParseFromString(file_in.read())
        except IOError:
            print("File at [" + map_path + "] is not accessible")
            exit()

    def _load_traffic_light(self):
        for signal in self.hd_map.signal:
            self.signal_dict[signal.id.id] = signal

    def _load_lane(self):
        for lane in self.hd_map.lane:
            self.lane_dict[lane.id.id] = lane

    def _load_overlap(self):
        for overlap in self.hd_map.overlap:
            self.overlap_dict[overlap.id.id] = overlap

    def _get_traffic_light_by_id(self, id):
        return self.signal_dict[id]

    def _get_lane_by_id(self, id):
        return self.lane_dict[id]

    def _get_overlap_by_id(self, id):
        return self.overlap_dict[id]

    def draw_traffic_lights(self, center_x, center_y, center_heading, observed_traffic_lights, coordinate_heading=0.):
        local_map = np.zeros(
            [self.GRID[1], self.GRID[0], 1], dtype=np.uint8)
        self.local_base_point = np.array([center_x, center_y])
        self.local_base_heading = center_heading

        for traffic_light_status in observed_traffic_lights:
            traffic_light = self._get_traffic_light_by_id(traffic_light_status.id)
            traffic_light_color = (255)
            if traffic_light_status.color == traffic_light_detection_pb2.TrafficLight.RED:
                traffic_light_color = (255)
            elif traffic_light_status.color == traffic_light_detection_pb2.TrafficLight.YELLOW:
                traffic_light_color = (192)
            else:
                traffic_light_color = (128)

            for overlap_id in traffic_light.overlap_id:
                overlap = self._get_overlap_by_id(overlap_id.id)
                for overlap_object in overlap.object:
                    if overlap_object.HasField("lane_overlap_info"):
                        lane = self._get_lane_by_id(overlap_object.id.id)
                        for segment in lane.central_curve.segment:
                            for i in range(len(segment.line_segment.point) - 1):
                                p0 = tuple(renderer_utils.get_img_idx(
                                    renderer_utils.point_affine_transformation(
                                        np.array([segment.line_segment.point[i].x,
                                                segment.line_segment.point[i].y]),
                                        self.local_base_point,
                                        np.pi / 2 - self.local_base_heading + coordinate_heading),
                                    self.local_base_point_idx,
                                    self.resolution))
                                p1 = tuple(renderer_utils.get_img_idx(
                                    renderer_utils.point_affine_transformation(
                                        np.array([segment.line_segment.point[i + 1].x,
                                                segment.line_segment.point[i + 1].y]),
                                        self.local_base_point,
                                        np.pi / 2 - self.local_base_heading + coordinate_heading),
                                    self.local_base_point_idx,
                                    self.resolution))
                                cv.line(local_map, tuple(p0), tuple(p1),
                                        color=traffic_light_color, thickness=4)
        return local_map


if __name__ == "__main__":
    config_file = '/fuel/fueling/planning/datasets/semantic_map_feature/planning_semantic_map_config.pb.txt'
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.55.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    output_dir = './data_traffic_light/'
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists, delete it!")
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("Making output directory: " + output_dir)

    ego_pos_dict = dict()
    traffic_lights_mapping = TrafficLightsImgRenderer(config_file, "sunnyvale_with_two_offices")
    for frame in offline_frames.learning_data:
        img = traffic_lights_mapping.draw_traffic_lights(
            frame.localization.position.x, frame.localization.position.y, frame.localization.heading, frame.traffic_light_detection.traffic_light)
        key = "{}@{:.3f}".format(frame.frame_num, frame.timestamp_sec)
        filename = key + ".png"
        ego_pos_dict[key] = [frame.localization.position.x,
                             frame.localization.position.y, frame.localization.heading]
        cv.imwrite(os.path.join(output_dir, filename), img)
    np.save(os.path.join(output_dir + "/ego_pos.npy"), ego_pos_dict)
