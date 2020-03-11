#!/usr/bin/env python
import os
import shutil

import numpy as np
import cv2 as cv

from learning_algorithms.planning.data_preprocessing.semantic_map_feature.agent_box_img_renderer import AgentBoxImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.agent_poses_history_img_renderer import AgentPosesHistoryImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.base_roadmap_img_renderer import BaseRoadMapImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.base_speedlimit_img_renderer import BaseSpeedLimitImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.obstacles_img_renderer import ObstaclesImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.roadmap_img_renderer import RoadMapImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.routing_img_renderer import RoutingImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.speed_limit_img_renderer import SpeedLimitImgRenderer
from learning_algorithms.planning.data_preprocessing.semantic_map_feature.traffic_lights_img_renderer import TrafficLightsImgRenderer
from modules.planning.proto import learning_data_pb2


class ChauffeurNetFeatureGenerator(object):
    """class of ChauffeurNetFeatureGenerator to initialize renderers and aggregate input features"""

    def __init__(self, region):
        # Draw base maps
        # self.base_road_map_mapping = BaseRoadMapImgRenderer(region)
        # self.base_speed_limit_mapping = BaseSpeedLimitImgRenderer(region)
        # self._dump_base_map()

        self.agent_box_mapping = AgentBoxImgRenderer()
        self.agent_pose_history_mapping = AgentPosesHistoryImgRenderer()
        self.obstacles_mapping = ObstaclesImgRenderer()
        self.road_map_mapping = RoadMapImgRenderer(region)
        self.routing_mapping = RoutingImgRenderer(region)
        self.speed_limit_mapping = SpeedLimitImgRenderer(region)
        self.traffic_lights_mapping = TrafficLightsImgRenderer(region)

    def _dump_base_map(self):
        cv.imwrite(self.base_road_map_mapping.region + ".png",
                   self.base_road_map_mapping.base_map)
        cv.imwrite(self.base_speed_limit_mapping.region + ".png",
                   self.base_speed_limit_mapping.base_map)

    def draw_features(self, frame_num, frame_time_sec, ego_pose_history, obstacle,
                      center_x, center_y, center_heading, routing_response,
                      observed_traffic_lights, output_dirs):
        agent_box_img = self.agent_box_mapping.draw_agent_box()
        agent_pose_history_img = self.agent_pose_history_mapping.draw_agent_poses_history(
            ego_pose_history)
        obstacles_img = self.obstacles_mapping.draw_obstacles(
            frame_time_sec, obstacle)
        road_map_img = self.road_map_mapping.draw_roadmap(
            center_x, center_y, center_heading)
        routing_img = self.routing_mapping.draw_routing(
            center_x, center_y, center_heading, routing_response)
        speed_limit_img = self.speed_limit_mapping.draw_speedlimit(
            center_x, center_y, center_heading)
        traffic_lights_img = self.traffic_lights_mapping.draw_traffic_lights(
            center_x, center_y, center_heading, observed_traffic_lights)
        imgs_list = [agent_box_img, agent_pose_history_img, obstacles_img,
                     road_map_img, routing_img, speed_limit_img, traffic_lights_img]
        for i in range(0, len(output_dirs), 1):
            key = "{}@{:.3f}".format(frame_num, frame_time_sec)
            filename = key + ".png"
            cv.imwrite(os.path.join(output_dirs[i], filename), imgs_list[i])

    def aggregate_features(self):
        pass


if __name__ == "__main__":
    offline_frames = learning_data_pb2.LearningData()
    with open("/apollo/data/learning_data.31.bin", 'rb') as file_in:
        offline_frames.ParseFromString(file_in.read())
    print("Finish reading proto...")

    region = "sunnyvale_with_two_offices"
    chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(region)
    print("Finish loading chauffeur_net_feature_generator...")

    output_dirs = ['./data_local_agent_box/', './data_agent_pose_history/', './data_obstacles/',
                   './data_local_road_map/', './data_local_routing/', './data_local_speed_limit/', './data_traffic_light/']
    for output_dir in output_dirs:
        if os.path.isdir(output_dir):
            print(output_dir + " directory exists, delete it!")
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        print("Making output directory: " + output_dir)

    for frame in offline_frames.learning_data:
        ego_pose_history = []
        for past_pose in frame.adc_trajectory_point:
            ego_pose_history.append(
                [past_pose.trajectory_point.path_point.x,
                 past_pose.trajectory_point.path_point.y,
                 past_pose.trajectory_point.path_point.theta])
        chauffeur_net_feature_generator.draw_features(frame.frame_num,
                                                      frame.timestamp_sec,
                                                      ego_pose_history,
                                                      frame.obstacle,
                                                      frame.localization.position.x,
                                                      frame.localization.position.y,
                                                      frame.localization.heading,
                                                      frame.routing_response.lane_id,
                                                      frame.traffic_light,
                                                      output_dirs)
        print(frame.frame_num)        
