#!/usr/bin/env python
import os
import shutil

import numpy as np
import cv2 as cv

from modules.map.proto import map_pb2
from modules.planning.proto import learning_data_pb2

import fueling.common.logging as logging
from fueling.planning.input_feature_preprocessor.agent_box_img_renderer \
    import AgentBoxImgRenderer
from fueling.planning.input_feature_preprocessor.agent_poses_future_img_renderer \
    import AgentPosesFutureImgRenderer
from fueling.planning.input_feature_preprocessor.agent_poses_history_img_renderer \
    import AgentPosesHistoryImgRenderer
from fueling.planning.input_feature_preprocessor.base_offroad_mask_img_renderer \
    import BaseOffroadMaskImgRenderer
from fueling.planning.input_feature_preprocessor.base_roadmap_img_renderer \
    import BaseRoadMapImgRenderer
from fueling.planning.input_feature_preprocessor.base_speedlimit_img_renderer \
    import BaseSpeedLimitImgRenderer
from fueling.planning.input_feature_preprocessor.obstacle_history_img_renderer \
    import ObstacleHistoryImgRenderer
from fueling.planning.input_feature_preprocessor.obstacle_predictions_img_renderer \
    import ObstaclePredictionsImgRenderer
from fueling.planning.input_feature_preprocessor.offroad_mask_img_renderer \
    import OffroadMaskImgRenderer
from fueling.planning.input_feature_preprocessor.roadmap_img_renderer \
    import RoadMapImgRenderer
from fueling.planning.input_feature_preprocessor.routing_img_renderer \
    import RoutingImgRenderer
from fueling.planning.input_feature_preprocessor.speed_limit_img_renderer \
    import SpeedLimitImgRenderer
from fueling.planning.input_feature_preprocessor.traffic_lights_img_renderer \
    import TrafficLightsImgRenderer
import fueling.planning.input_feature_preprocessor.renderer_utils as renderer_utils


class ChauffeurNetFeatureGenerator(object):
    """class of ChauffeurNetFeatureGenerator to initialize renderers and aggregate input features"""

    def __init__(self, regions_list, config_file, base_map_img_dir, base_map_data_dir):
        self.base_map_img_dir = base_map_img_dir
        self.config_file = config_file

        self.base_maps_check(regions_list, base_map_img_dir)

        self.agent_box_mapping = AgentBoxImgRenderer(self.config_file)
        self.agent_pose_future_mapping = AgentPosesFutureImgRenderer(
            self.config_file)
        self.agent_pose_history_mapping = AgentPosesHistoryImgRenderer(
            self.config_file)
        self.obstacle_history_mapping = ObstacleHistoryImgRenderer(
            self.config_file)
        self.obstacle_predictions_mapping = ObstaclePredictionsImgRenderer(
            self.config_file)

        self.region_base_offroad_img_dict = self.load_base_offroad_imgs(regions_list,
                                                                        base_map_img_dir)
        self.region_base_roadmap_img_dict = self.load_base_roadmap_imgs(regions_list,
                                                                        base_map_img_dir)
        self.region_base_speedlimit_img_dict = self.load_base_speedlimit_imgs(regions_list,
                                                                              base_map_img_dir)
        self.region_base_map_data_dict = self.load_base_map_data(regions_list,
                                                                 base_map_data_dir)
        self.offroad_mask_mapping = OffroadMaskImgRenderer(
            self.config_file, self.region_base_offroad_img_dict)
        self.road_map_mapping = RoadMapImgRenderer(
            self.config_file, self.region_base_roadmap_img_dict)
        self.routing_mapping = RoutingImgRenderer(
            self.config_file, self.region_base_map_data_dict)
        self.speed_limit_mapping = SpeedLimitImgRenderer(
            self.config_file, self.region_base_speedlimit_img_dict)
        self.traffic_lights_mapping = TrafficLightsImgRenderer(
            self.config_file, self.region_base_map_data_dict)

    def load_base_offroad_imgs(self, regions_list, base_map_img_dir):
        region_base_offroad_img_dict = {}
        for region in regions_list:
            offroad_img = cv.imread(os.path.join(
                base_map_img_dir, region + "_offroad_mask.png"),
                cv.IMREAD_UNCHANGED)
            region_base_offroad_img_dict[region] = offroad_img
        return region_base_offroad_img_dict

    def load_base_roadmap_imgs(self, regions_list, base_map_img_dir):
        region_base_roadmap_img_dict = {}
        for region in regions_list:
            roadmap_img = cv.imread(os.path.join(
                base_map_img_dir, region + ".png"))
            region_base_roadmap_img_dict[region] = roadmap_img
        return region_base_roadmap_img_dict

    def load_base_speedlimit_imgs(self, regions_list, base_map_img_dir):
        region_base_speedlimit_img_dict = {}
        for region in regions_list:
            speedlimit_img = cv.imread(os.path.join(
                base_map_img_dir, region + "_speedlimit.png"))
            region_base_speedlimit_img_dict[region] = speedlimit_img
        return region_base_speedlimit_img_dict

    def load_base_map_data(self, regions_list, base_map_data_dir):
        region_base_map_data_dict = {}
        for region in regions_list:
            hd_map = map_pb2.Map()
            map_path = os.path.join(
                base_map_data_dir, region + "/sim_map.bin")
            try:
                with open(map_path, 'rb') as file_in:
                    hd_map.ParseFromString(file_in.read())
            except IOError:
                logging.error("File at [" + map_path + "] is not accessible")
                exit()
            region_base_map_data_dict[region] = hd_map
        return region_base_map_data_dict

    def base_maps_check(self, regions_list, base_map_img_dir):
        for region in regions_list:
            if not os.path.isfile(os.path.join(base_map_img_dir,
                                               region + ".png")) \
                    or not os.path.isfile(os.path.join(base_map_img_dir,
                                                       region + "_speedlimit.png")) \
                    or not os.path.isfile(os.path.join(base_map_img_dir,
                                                       region + "_offroad_mask.png")):
                logging.error("a base map img is not accessible")
                exit()

    @staticmethod
    def draw_base_map(regions_list, config_file, base_map_img_dir, base_map_data_dir):
        for region in regions_list:
            base_offroad_mask_mapping = BaseOffroadMaskImgRenderer(
                config_file, region, base_map_data_dir)
            base_road_map_mapping = BaseRoadMapImgRenderer(
                config_file, region, base_map_data_dir)
            base_speed_limit_mapping = BaseSpeedLimitImgRenderer(
                config_file, region, base_map_data_dir)
            cv.imwrite(os.path.join(base_map_img_dir, region + "_offroad_mask.png"),
                       base_offroad_mask_mapping.base_map)
            cv.imwrite(os.path.join(base_map_img_dir, region + ".png"),
                       base_road_map_mapping.base_map)
            cv.imwrite(os.path.join(base_map_img_dir, region + "_speedlimit.png"),
                       base_speed_limit_mapping.base_map)

    def render_seperated_img_features(self, region, frame_num, frame_time_sec, ego_pose_history,
                                      obstacle, center_x, center_y, center_heading,
                                      routing_response, observed_traffic_lights, output_dirs,
                                      coordinate_heading=0., past_motion_dropout=False):
        '''
        For debug purposes, features are drawn seprately
        agent_box_img: 1 channel np.array image
        agent_pose_history_img: 1 channel np.array image
        obstacle_history_img: 1 channel np.array image
        obstacle_predictions_img: 1 channel np.array image
        road_map_img: 3 channel np.array image
        routing_img: 1 channel np.array image
        speed_limit_img: 3 channel np.array image
        traffic_lights_img: 1 channel np.array image
        '''
        agent_box_img = self.agent_box_mapping.draw_agent_box(
            coordinate_heading)
        agent_pose_history_img = self.agent_pose_history_mapping.draw_agent_poses_history(
            frame_time_sec,
            center_x,
            center_y,
            center_heading,
            ego_pose_history,
            coordinate_heading,
            past_motion_dropout)
        obstacle_history_img = self.obstacle_history_mapping.draw_obstacle_history(
            frame_time_sec,
            obstacle,
            coordinate_heading)
        obstacle_predictions_img = self.obstacle_predictions_mapping.draw_obstacle_box_prediction(
            frame_time_sec,
            obstacle,
            coordinate_heading)
        road_map_img = self.road_map_mapping.draw_roadmap(
            region,
            center_x,
            center_y,
            center_heading,
            coordinate_heading)
        routing_img = self.routing_mapping.draw_local_routing(
            region,
            center_x,
            center_y,
            center_heading,
            routing_response,
            coordinate_heading)
        speed_limit_img = self.speed_limit_mapping.draw_speedlimit(
            region,
            center_x,
            center_y,
            center_heading,
            coordinate_heading)
        traffic_lights_img = self.traffic_lights_mapping.draw_traffic_lights(
            region,
            center_x,
            center_y,
            center_heading,
            observed_traffic_lights,
            coordinate_heading)
        imgs_list = [agent_box_img, agent_pose_history_img,
                     obstacle_history_img, obstacle_predictions_img,
                     road_map_img, routing_img, speed_limit_img, traffic_lights_img]
        for i in range(len(output_dirs)):
            key = "{}@{:.3f}".format(frame_num, frame_time_sec)
            filename = key + ".png"
            cv.imwrite(os.path.join(output_dirs[i], filename), imgs_list[i])

    def render_stacked_img_features(self, region, frame_time_sec, ego_pose_history,
                                    obstacle, center_x, center_y, center_heading,
                                    routing_response, observed_traffic_lights,
                                    coordinate_heading=0., past_motion_dropout=False):
        '''
        agent_box_img: 1 channel np.array image, [0]
        agent_pose_history_img: 1 channel np.array image, [1]
        obstacle_history_img: 1 channel np.array image, [2]
        obstacle_predictions_img: 1 channel np.array image, [3]
        road_map_img: 3 channel np.array image, [4:7]
        routing_img: 1 channel np.array image, [7]
        speed_limit_img: 3 channel np.array image, [8:11]
        traffic_lights_img: 1 channel np.array image, [11]

        All images in np.unit8 and concatenated along channel axis
        '''

        agent_box_img = self.agent_box_mapping.draw_agent_box(
            coordinate_heading)
        agent_pose_history_img = self.agent_pose_history_mapping.draw_agent_poses_history(
            frame_time_sec,
            center_x,
            center_y,
            center_heading,
            ego_pose_history,
            coordinate_heading,
            past_motion_dropout)
        obstacle_history_img = self.obstacle_history_mapping.draw_obstacle_history(
            frame_time_sec,
            obstacle,
            coordinate_heading)
        obstacle_predictions_img = self.obstacle_predictions_mapping.draw_obstacle_box_prediction(
            frame_time_sec,
            obstacle,
            coordinate_heading)
        road_map_img = self.road_map_mapping.draw_roadmap(
            region,
            center_x,
            center_y,
            center_heading,
            coordinate_heading)
        routing_img = self.routing_mapping.draw_local_routing(
            region,
            center_x,
            center_y,
            center_heading,
            routing_response,
            coordinate_heading)
        speed_limit_img = self.speed_limit_mapping.draw_speedlimit(
            region,
            center_x,
            center_y,
            center_heading,
            coordinate_heading)
        traffic_lights_img = self.traffic_lights_mapping.draw_traffic_lights(
            region,
            center_x,
            center_y,
            center_heading,
            observed_traffic_lights,
            coordinate_heading)

        return np.concatenate([agent_box_img, agent_pose_history_img,
                               obstacle_history_img, obstacle_predictions_img,
                               road_map_img, routing_img, speed_limit_img,
                               traffic_lights_img], axis=2)

    def render_merged_img_feature(self, stacked_img_features):
        '''
        Merging sequence would be as follow up to down:
        road_map_img: 3 channel np.array image
        routing_img: 1 channel np.array image
        speed_limit_img: 3 channel np.array image
        traffic_lights_img: 1 channel np.array image
        obstacle_history_img: 1 channel np.array image
        obstacle_predictions_img: 1 channel np.array image
        agent_box_img: 1 channel np.array image
        agent_pose_history_img: 1 channel np.array image
        '''
        road_map_img = stacked_img_features[:, :, 4:7]
        routing_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 7], axis=2), 3, axis=2)
        # speed_limit_img = stacked_img_features[:, :, 8:11]
        traffic_lights_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 11], 2), 3, axis=2)
        # draw obstacle past in red color
        obstacle_history_img = renderer_utils.img_white_gradient_to_color_gradient(
            np.repeat(np.expand_dims(
                stacked_img_features[:, :, 2],
                axis=2),
                3, axis=2),
            (0, 0, 255))
        # draw obstacle future in green color
        obstacle_predictions_img = renderer_utils.img_white_gradient_to_color_gradient(
            np.repeat(np.expand_dims(
                stacked_img_features[:, :, 3],
                axis=2),
                3, axis=2),
            (0, 255, 0))
        agent_box_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 0], axis=2), 3, axis=2)
        agent_pose_history_img = np.repeat(np.expand_dims(
            stacked_img_features[:, :, 1], axis=2), 3, axis=2)
        merged_img = renderer_utils.img_notblack_stacking(
            routing_img, road_map_img)
        # merged_img = renderer_utils.img_notblack_stacking(
        #     speed_limit_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            traffic_lights_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            obstacle_history_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            obstacle_predictions_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            agent_box_img, merged_img)
        merged_img = renderer_utils.img_notblack_stacking(
            agent_pose_history_img, merged_img)
        return merged_img

    def render_gt_pose_dist(self, center_x, center_y, center_heading,
                            ego_pose_future, timestamp_idx, coordinate_heading=0.):
        return self.agent_pose_future_mapping.draw_agent_pose_future(center_x,
                                                                     center_y,
                                                                     center_heading,
                                                                     ego_pose_future,
                                                                     timestamp_idx,
                                                                     coordinate_heading)

    def render_gt_box(self, center_x, center_y, center_heading,
                      ego_pose_future, timestamp_idx, coordinate_heading=0.):
        return self.agent_pose_future_mapping.draw_agent_box_future(center_x,
                                                                    center_y,
                                                                    center_heading,
                                                                    ego_pose_future,
                                                                    timestamp_idx,
                                                                    coordinate_heading)

    def render_offroad_mask(self, region, center_x, center_y, center_heading,
                            coordinate_heading=0.):
        return self.offroad_mask_mapping.draw_offroad_mask(region,
                                                           center_x,
                                                           center_y,
                                                           center_heading,
                                                           coordinate_heading)

    def render_routing_mask(self, region, center_x, center_y, center_heading,
                            routing_response, coordinate_heading=0.):
        return self.routing_mapping.draw_constant_coloring_local_routing_mask(region,
                                                                              center_x,
                                                                              center_y,
                                                                              center_heading,
                                                                              routing_response,
                                                                              coordinate_heading)

    def render_obstacle_box_prediction_frame(
            self, center_x, center_y, center_heading,
            obstacles, timestamp_idx, coordinate_heading=0.):
        return self.obstacle_predictions_mapping.draw_obstacle_box_prediction_frame(
            obstacles,
            timestamp_idx,
            coordinate_heading)

    def render_initial_agent_states(self, coordinate_heading=0.):
        agent_box_img = self.agent_box_mapping.draw_agent_box(
            coordinate_heading)
        agent_pose_img = self.agent_pose_history_mapping.draw_agent_current_pose()
        return agent_box_img, agent_pose_img
