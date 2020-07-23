import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from modules.planning.proto import planning_semantic_map_config_pb2
from modules.planning.proto import learning_data_pb2

from fueling.common.coord_utils import CoordUtils
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils
from fueling.planning.input_feature_preprocessor.chauffeur_net_feature_generator \
    import ChauffeurNetFeatureGenerator


class TrajectoryImitationCNNFCDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir, map_path, region,
                 img_feature_rotation=False, past_motion_dropout=False,
                 ouput_point_num=10, evaluate_mode=False):
        # TODO(Jinyun): refine transform function
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])

        logging.info('Processing directory: {}'.format(data_dir))
        self.instances = file_utils.list_files(data_dir)

        # TODO(Jinyun): add multi-map support
        # region = "sunnyvale_with_two_offices"

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(renderer_config_file,
                                                                            imgs_dir,
                                                                            region, map_path)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file, renderer_config)
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        self.ouput_point_num = ouput_point_num
        self.evaluate_mode = evaluate_mode

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame_name = self.instances[idx]

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = torch.rand(
                1) * 2 * self.max_rand_coordinate_heading - self.max_rand_coordinate_heading

        is_past_motion_dropout = False
        if self.past_motion_dropout:
            is_past_motion_dropout = torch.rand(1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(frame.frame_num,
                                        frame.adc_trajectory_point[-1].timestamp_sec,
                                        frame.adc_trajectory_point,
                                        frame.obstacle,
                                        current_x,
                                        current_y,
                                        current_theta,
                                        frame.routing.local_routing_lane_id,
                                        frame.traffic_light_detection.traffic_light,
                                        coordinate_heading,
                                        is_past_motion_dropout)
        transformed_img_feature = self.img_transform(img_feature)

        ref_coords = [current_x,
                      current_y,
                      current_theta]
        pred_points = np.zeros((0, 4))
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            if i + 1 > self.ouput_point_num:
                break
            # TODO(Jinyun): evaluate whether use heading and acceleration
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = pred_theta - ref_coords[2]
            pred_v = pred_point.trajectory_point.v
            pred_points = np.vstack((pred_points, np.asarray(
                [local_coords[0], local_coords[1], heading_diff, pred_v])))

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < self.ouput_point_num:
            return self.__getitem__(idx - 1)

        if self.evaluate_mode:
            merged_img_feature = self.chauffeur_net_feature_generator.render_merged_img_feature(
                img_feature)
            return (transformed_img_feature,
                    torch.from_numpy(pred_points).float(),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return (transformed_img_feature, torch.from_numpy(pred_points).float())


class TrajectoryImitationConvRNNDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir, map_path, region,
                 img_feature_rotation=False, past_motion_dropout=False,
                 ouput_point_num=10, evaluate_mode=False):
        # TODO(Jinyun): refine transform function
        self.img_feature_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
        self.img_bitmap_transform = transforms.Compose([
            # Normalized to [0, 1]
            transforms.ToTensor()])

        logging.info('Processing directory: {}'.format(data_dir))
        self.instances = file_utils.list_files(data_dir)

        # TODO(Jinyun): add multi-map support
        # region = "sunnyvale_with_two_offices"

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(renderer_config_file,
                                         imgs_dir,
                                         region,
                                         map_path,
                                         base_map_update_flag=False)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file, renderer_config)
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        self.img_size = [renderer_config.width, renderer_config.height]
        self.ouput_point_num = ouput_point_num
        self.evaluate_mode = evaluate_mode

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame_name = self.instances[idx]

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = torch.rand(
                1) * 2 * self.max_rand_coordinate_heading - self.max_rand_coordinate_heading

        is_past_motion_dropout = False
        if self.past_motion_dropout:
            is_past_motion_dropout = torch.rand(1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(frame.frame_num,
                                        frame.adc_trajectory_point[-1].timestamp_sec,
                                        frame.adc_trajectory_point,
                                        frame.obstacle,
                                        current_x,
                                        current_y,
                                        current_theta,
                                        frame.routing.local_routing_lane_id,
                                        frame.traffic_light_detection.traffic_light,
                                        coordinate_heading,
                                        is_past_motion_dropout)
        transformed_img_feature = self.img_feature_transform(img_feature)

        offroad_mask = self.chauffeur_net_feature_generator.\
            render_offroad_mask(current_x,
                                current_y,
                                current_theta,
                                coordinate_heading)
        offroad_mask = self.img_bitmap_transform(offroad_mask)
        offroad_mask = offroad_mask.repeat(self.ouput_point_num, 1, 1, 1)

        ref_coords = [current_x,
                      current_y,
                      current_theta]
        pred_points = np.zeros((0, 4))
        pred_pose_dists = torch.rand(
            self.ouput_point_num, 1, self.img_size[1], self.img_size[0])
        pred_boxs = torch.rand(self.ouput_point_num, 1,
                               self.img_size[1], self.img_size[0])
        pred_obs = torch.rand(self.ouput_point_num, 1,
                              self.img_size[1], self.img_size[0])
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            if i + 1 > self.ouput_point_num:
                break

            # TODO(Jinyun): evaluate whether use heading and acceleration
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            pred_v = pred_point.trajectory_point.v
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = pred_theta - ref_coords[2]
            pred_points = np.vstack((pred_points, np.asarray(
                [local_coords[0], local_coords[1], heading_diff, pred_v])))

            gt_pose_dist = self.chauffeur_net_feature_generator.\
                render_gt_pose_dist(current_x,
                                    current_y,
                                    current_theta,
                                    frame.output.adc_future_trajectory_point,
                                    i,
                                    coordinate_heading)
            gt_pose_dist = self.img_bitmap_transform(gt_pose_dist)
            pred_pose_dists[i, :, :, :] = gt_pose_dist

            gt_pose_box = self.chauffeur_net_feature_generator.\
                render_gt_box(current_x,
                              current_y,
                              current_theta,
                              frame.output.adc_future_trajectory_point,
                              i,
                              coordinate_heading)
            gt_pose_box = self.img_bitmap_transform(gt_pose_box)
            pred_boxs[i, :, :, :] = gt_pose_box

            pred_obs_box = self.chauffeur_net_feature_generator.\
                render_obstacle_box_prediction_frame(current_x,
                                                     current_y,
                                                     current_theta,
                                                     frame.obstacle,
                                                     i,
                                                     coordinate_heading)
            pred_obs_box = self.img_bitmap_transform(pred_obs_box)
            pred_obs[i, :, :, :] = pred_obs_box

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < self.ouput_point_num:
            return self.__getitem__(idx - 1)

        # draw agent current pose and box for hidden state intialization
        agent_current_box_img, agent_current_pose_img = self.chauffeur_net_feature_generator.\
            render_initial_agent_states(coordinate_heading)
        if self.img_bitmap_transform:
            agent_current_box_img = self.img_bitmap_transform(
                agent_current_box_img)
            agent_current_pose_img = self.img_bitmap_transform(
                agent_current_pose_img)

        if self.evaluate_mode:
            merged_img_feature = self.chauffeur_net_feature_generator.render_merged_img_feature(
                img_feature)
            return ((transformed_img_feature,
                     agent_current_pose_img,
                     agent_current_box_img),
                    (pred_pose_dists,
                     pred_boxs,
                     torch.from_numpy(pred_points).float(),
                     pred_obs,
                     offroad_mask),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return ((transformed_img_feature,
                 agent_current_pose_img,
                 agent_current_box_img),
                (pred_pose_dists,
                 pred_boxs,
                 torch.from_numpy(pred_points).float(),
                 pred_obs,
                 offroad_mask))


class TrajectoryImitationCNNLSTMDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir, map_path, region,
                 img_feature_rotation=False, past_motion_dropout=False,
                 history_point_num=10, ouput_point_num=10, evaluate_mode=False):
        # TODO(Jinyun): refine transform function
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])

        logging.info('Processing directory: {}'.format(data_dir))
        self.instances = file_utils.list_files(data_dir)

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(renderer_config_file,
                                                                            imgs_dir,
                                                                            region, map_path)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file, renderer_config)
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        self.history_point_num = history_point_num
        self.ouput_point_num = ouput_point_num
        self.evaluate_mode = evaluate_mode

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame_name = self.instances[idx]

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = torch.rand(
                1) * 2 * self.max_rand_coordinate_heading - self.max_rand_coordinate_heading

        is_past_motion_dropout = False
        if self.past_motion_dropout:
            is_past_motion_dropout = torch.rand(1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(frame.frame_num,
                                        frame.adc_trajectory_point[-1].timestamp_sec,
                                        frame.adc_trajectory_point,
                                        frame.obstacle,
                                        current_x,
                                        current_y,
                                        current_theta,
                                        frame.routing.local_routing_lane_id,
                                        frame.traffic_light_detection.traffic_light,
                                        coordinate_heading,
                                        is_past_motion_dropout)
        transformed_img_feature = self.img_transform(img_feature)

        ref_coords = [current_x,
                      current_y,
                      current_theta]

        hist_points = np.zeros((0, 4))
        for i, hist_point in enumerate(reversed(frame.adc_trajectory_point)):
            if i + 1 > self.history_point_num:
                break
            # TODO(Jinyun): evaluate whether use heading and acceleration
            hist_x = hist_point.trajectory_point.path_point.x
            hist_y = hist_point.trajectory_point.path_point.y
            hist_theta = hist_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [hist_x, hist_y], ref_coords)
            heading_diff = hist_theta - ref_coords[2]
            hist_v = hist_point.trajectory_point.v
            hist_points = np.vstack((np.asarray(
                [local_coords[0], local_coords[1], heading_diff, hist_v]), hist_points))

        # It only for the situation when car just started without too much history.
        while hist_points.shape[0] < self.history_point_num:
            hist_points = np.vstack((hist_points[0], hist_points))

        hist_points_step = np.zeros_like(hist_points)
        hist_points_step[1:, :] = hist_points[1:, :] - hist_points[:-1, :]

        pred_points = np.zeros((0, 4))
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            if i + 1 > self.ouput_point_num:
                break
            # TODO(Jinyun): evaluate whether use heading and acceleration
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = pred_theta - ref_coords[2]
            pred_v = pred_point.trajectory_point.v
            pred_points = np.vstack((pred_points, np.asarray(
                [local_coords[0], local_coords[1], heading_diff, pred_v])))

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < self.ouput_point_num:
            return self.__getitem__(idx - 1)

        if self.evaluate_mode:
            merged_img_feature = self.chauffeur_net_feature_generator.render_merged_img_feature(
                img_feature)
            return ((transformed_img_feature,
                     torch.from_numpy(hist_points).float(),
                     torch.from_numpy(hist_points_step).float()),
                    torch.from_numpy(pred_points).float(),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return ((transformed_img_feature,
                 torch.from_numpy(hist_points).float(),
                 torch.from_numpy(hist_points_step).float()),
                torch.from_numpy(pred_points).float())


class TrajectoryImitationCNNLSTMWithAENDataset(Dataset):
    def __init__(self, data_dir, renderer_config_file, imgs_dir, map_path, region,
                 img_feature_rotation=False, past_motion_dropout=False,
                 history_point_num=10, ouput_point_num=10, evaluate_mode=False):
        # TODO(Jinyun): refine transform function
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # 12 channels is used
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5])])
        self.img_bitmap_transform = transforms.Compose([
            # Normalized to [0, 1]
            transforms.ToTensor()])

        logging.info('Processing directory: {}'.format(data_dir))
        self.instances = file_utils.list_files(data_dir)

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        # TODO(Jinyun): recognize map_name in __getitem__
        self.chauffeur_net_feature_generator = ChauffeurNetFeatureGenerator(renderer_config_file,
                                                                            imgs_dir,
                                                                            region, map_path)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = planning_semantic_map_config_pb2.PlanningSemanticMapConfig()
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file, renderer_config)
        self.img_size = [renderer_config.width, renderer_config.height]
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        self.history_point_num = history_point_num
        self.ouput_point_num = ouput_point_num
        self.evaluate_mode = evaluate_mode

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame_name = self.instances[idx]

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = torch.rand(
                1) * 2 * self.max_rand_coordinate_heading - self.max_rand_coordinate_heading

        is_past_motion_dropout = False
        if self.past_motion_dropout:
            is_past_motion_dropout = torch.rand(1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(frame.frame_num,
                                        frame.adc_trajectory_point[-1].timestamp_sec,
                                        frame.adc_trajectory_point,
                                        frame.obstacle,
                                        current_x,
                                        current_y,
                                        current_theta,
                                        frame.routing.local_routing_lane_id,
                                        frame.traffic_light_detection.traffic_light,
                                        coordinate_heading,
                                        is_past_motion_dropout)
        transformed_img_feature = self.img_transform(img_feature)

        offroad_mask = self.chauffeur_net_feature_generator.\
            render_offroad_mask(current_x,
                                current_y,
                                current_theta,
                                coordinate_heading)
        offroad_mask = self.img_bitmap_transform(offroad_mask)
        offroad_mask = offroad_mask.repeat(self.ouput_point_num, 1, 1, 1)

        ref_coords = [current_x,
                      current_y,
                      current_theta]

        hist_points = np.zeros((0, 4))
        for i, hist_point in enumerate(reversed(frame.adc_trajectory_point)):
            if i + 1 > self.history_point_num:
                break
            # TODO(Jinyun): evaluate whether use heading and acceleration
            hist_x = hist_point.trajectory_point.path_point.x
            hist_y = hist_point.trajectory_point.path_point.y
            hist_theta = hist_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [hist_x, hist_y], ref_coords)
            heading_diff = hist_theta - ref_coords[2]
            hist_v = hist_point.trajectory_point.v
            hist_points = np.vstack((np.asarray(
                [local_coords[0], local_coords[1], heading_diff, hist_v]), hist_points))

        # It only for the situation when car just started without too much history.
        while hist_points.shape[0] < self.history_point_num:
            hist_points = np.vstack((hist_points[0], hist_points))

        hist_points_step = np.zeros_like(hist_points)
        hist_points_step[1:, :] = hist_points[1:, :] - hist_points[:-1, :]

        pred_boxs = torch.rand(self.ouput_point_num, 1,
                               self.img_size[1], self.img_size[0])
        pred_obs = torch.rand(self.ouput_point_num, 1,
                              self.img_size[1], self.img_size[0])
        pred_points = np.zeros((0, 4))
        for i, pred_point in enumerate(frame.output.adc_future_trajectory_point):
            if i + 1 > self.ouput_point_num:
                break

            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = pred_theta - ref_coords[2]
            pred_v = pred_point.trajectory_point.v
            pred_points = np.vstack((pred_points, np.asarray(
                [local_coords[0], local_coords[1], heading_diff, pred_v])))

            gt_pose_box = self.chauffeur_net_feature_generator.\
                render_gt_box(current_x,
                              current_y,
                              current_theta,
                              frame.output.adc_future_trajectory_point,
                              i,
                              coordinate_heading)
            gt_pose_box = self.img_bitmap_transform(gt_pose_box)
            pred_boxs[i, :, :, :] = gt_pose_box

            pred_obs_box = self.chauffeur_net_feature_generator.\
                render_obstacle_box_prediction_frame(current_x,
                                                     current_y,
                                                     current_theta,
                                                     frame.obstacle,
                                                     i,
                                                     coordinate_heading)
            pred_obs_box = self.img_bitmap_transform(pred_obs_box)
            pred_obs[i, :, :, :] = pred_obs_box

        # TODO(Jinyun): it's a tmp fix, will add data clean to make sure output point size is right
        if pred_points.shape[0] < self.ouput_point_num:
            return self.__getitem__(idx - 1)

        if self.evaluate_mode:
            merged_img_feature = self.chauffeur_net_feature_generator.render_merged_img_feature(
                img_feature)
            return ((transformed_img_feature,
                     torch.from_numpy(hist_points).float(),
                     torch.from_numpy(hist_points_step).float()),
                    torch.from_numpy(pred_points).float(),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return ((transformed_img_feature,
                 torch.from_numpy(hist_points).float(),
                 torch.from_numpy(hist_points_step).float()),
                (pred_boxs,
                 torch.from_numpy(pred_points).float(),
                 pred_obs,
                 offroad_mask))
