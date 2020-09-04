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
from fueling.planning.math_utils.math_utils import NormalizeAngle


class TrajectoryImitationCNNFCDataset(Dataset):
    def __init__(self, data_dir, regions_list, renderer_config_file,
                 renderer_base_map_img_dir,
                 renderer_base_map_data_dir,
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

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(regions_list,
                                         renderer_config_file,
                                         renderer_base_map_img_dir,
                                         renderer_base_map_data_dir)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file,
            planning_semantic_map_config_pb2.PlanningSemanticMapConfig())
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

        region = frame.map_name

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = np.random.uniform() * 2 * self.max_rand_coordinate_heading - \
                self.max_rand_coordinate_heading

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
            render_stacked_img_features(region,
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
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = NormalizeAngle(pred_theta - ref_coords[2])
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
    def __init__(self, data_dir, regions_list, renderer_config_file,
                 renderer_base_map_img_dir,
                 renderer_base_map_data_dir,
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

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(regions_list,
                                         renderer_config_file,
                                         renderer_base_map_img_dir,
                                         renderer_base_map_data_dir)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file,
            planning_semantic_map_config_pb2.PlanningSemanticMapConfig())
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

        region = frame.map_name

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = np.random.uniform() * 2 * self.max_rand_coordinate_heading - \
                self.max_rand_coordinate_heading

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
            render_stacked_img_features(region,
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
            render_offroad_mask(region,
                                current_x,
                                current_y,
                                current_theta,
                                coordinate_heading)
        offroad_mask = self.img_bitmap_transform(offroad_mask)
        offroad_mask = offroad_mask.repeat(self.ouput_point_num, 1, 1, 1)

        routing_mask = self.chauffeur_net_feature_generator.\
            render_routing_mask(region,
                                current_x,
                                current_y,
                                current_theta,
                                frame.routing.local_routing_lane_id,
                                coordinate_heading)
        routing_mask = self.img_bitmap_transform(routing_mask)
        routing_mask = routing_mask.repeat(self.ouput_point_num, 1, 1, 1)

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

            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            pred_v = pred_point.trajectory_point.v
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = NormalizeAngle(pred_theta - ref_coords[2])
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
                    (torch.from_numpy(pred_points).float(),
                     pred_boxs,
                     pred_pose_dists,
                     pred_obs,
                     offroad_mask),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return ((transformed_img_feature,
                 agent_current_pose_img,
                 agent_current_box_img),
                (torch.from_numpy(pred_points).float(),
                 pred_boxs,
                 pred_pose_dists,
                 pred_obs,
                 offroad_mask,
                 routing_mask))


class TrajectoryImitationCNNLSTMDataset(Dataset):
    def __init__(self, data_dir, regions_list, renderer_config_file,
                 renderer_base_map_img_dir,
                 renderer_base_map_data_dir,
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
        self.instances = list(filter(lambda frame_path:
                                     'all_is_synthesized' not in frame_path,
                                     file_utils.list_files(data_dir)))

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(regions_list,
                                         renderer_config_file,
                                         renderer_base_map_img_dir,
                                         renderer_base_map_data_dir)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file,
            planning_semantic_map_config_pb2.PlanningSemanticMapConfig())
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

        region = frame.map_name

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = np.random.uniform() * 2 * self.max_rand_coordinate_heading - \
                self.max_rand_coordinate_heading

        is_past_motion_dropout = False
        if self.past_motion_dropout:
            is_past_motion_dropout = torch.rand(1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_traj_point = frame.adc_trajectory_point[-1].trajectory_point
        current_path_point = current_traj_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta
        current_v = torch.tensor([current_traj_point.v]).float()

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(region,
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
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = NormalizeAngle(pred_theta - ref_coords[2])
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
                     current_v),
                    torch.from_numpy(pred_points).float(),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec,
                    current_x,
                    current_y,
                    current_theta,
                    current_v)

        return ((transformed_img_feature,
                 current_v),
                torch.from_numpy(pred_points).float())


class TrajectoryImitationSelfCNNLSTMDataset(Dataset):
    def __init__(self, data_dir, regions_list, renderer_config_file,
                 renderer_base_map_img_dir,
                 renderer_base_map_data_dir,
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

        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(regions_list,
                                         renderer_config_file,
                                         renderer_base_map_img_dir,
                                         renderer_base_map_data_dir)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file,
            planning_semantic_map_config_pb2.PlanningSemanticMapConfig())
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

        region = frame.map_name

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = np.random.uniform() * 2 * self.max_rand_coordinate_heading - \
                self.max_rand_coordinate_heading

        is_past_motion_dropout = False
        if self.past_motion_dropout:
            is_past_motion_dropout = torch.rand(1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_path_point = frame.adc_trajectory_point[-1].trajectory_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta
        current_v = frame.adc_trajectory_point[-1].trajectory_point.v

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(region,
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
            hist_x = hist_point.trajectory_point.path_point.x
            hist_y = hist_point.trajectory_point.path_point.y
            hist_theta = hist_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [hist_x, hist_y], ref_coords)
            heading_diff = NormalizeAngle(hist_theta - ref_coords[2])
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
            pred_x = pred_point.trajectory_point.path_point.x
            pred_y = pred_point.trajectory_point.path_point.y
            pred_theta = pred_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [pred_x, pred_y], ref_coords)
            heading_diff = NormalizeAngle(pred_theta - ref_coords[2])
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
                    frame.message_timestamp_sec,
                    current_x,
                    current_y,
                    current_theta,
                    current_v)

        return ((transformed_img_feature,
                 torch.from_numpy(hist_points).float(),
                 torch.from_numpy(hist_points_step).float()),
                torch.from_numpy(pred_points).float())


class TrajectoryImitationSelfCNNLSTMWithEnvLossDataset(Dataset):
    def __init__(self, data_dir, regions_list, renderer_config_file,
                 renderer_base_map_img_dir,
                 renderer_base_map_data_dir,
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

        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(regions_list,
                                         renderer_config_file,
                                         renderer_base_map_img_dir,
                                         renderer_base_map_data_dir)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file,
            planning_semantic_map_config_pb2.PlanningSemanticMapConfig())
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

        is_synthesized = 'is_synthesized' in frame_name
        is_turning = 'NO_TURN' not in frame_name

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        region = frame.map_name

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = np.random.uniform() * 2 * self.max_rand_coordinate_heading - \
                self.max_rand_coordinate_heading

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
            render_stacked_img_features(region,
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
            render_offroad_mask(region,
                                current_x,
                                current_y,
                                current_theta,
                                coordinate_heading)
        offroad_mask = self.img_bitmap_transform(offroad_mask)
        offroad_mask = offroad_mask.repeat(self.ouput_point_num, 1, 1, 1)

        routing_mask = self.chauffeur_net_feature_generator.\
            render_routing_mask(region,
                                current_x,
                                current_y,
                                current_theta,
                                frame.routing.local_routing_lane_id,
                                coordinate_heading)
        routing_mask = self.img_bitmap_transform(routing_mask)
        routing_mask = routing_mask.repeat(self.ouput_point_num, 1, 1, 1)

        ref_coords = [current_x,
                      current_y,
                      current_theta]

        hist_points = np.zeros((0, 4))
        for i, hist_point in enumerate(reversed(frame.adc_trajectory_point)):
            if i + 1 > self.history_point_num:
                break
            hist_x = hist_point.trajectory_point.path_point.x
            hist_y = hist_point.trajectory_point.path_point.y
            hist_theta = hist_point.trajectory_point.path_point.theta
            local_coords = CoordUtils.world_to_relative(
                [hist_x, hist_y], ref_coords)
            heading_diff = NormalizeAngle(hist_theta - ref_coords[2])
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
            heading_diff = NormalizeAngle(pred_theta - ref_coords[2])
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
                     torch.from_numpy(hist_points_step).float(),
                     coordinate_heading + np.pi / 2),
                    (torch.from_numpy(pred_points).float(),
                     pred_boxs,
                     torch.empty(1),
                     pred_obs,
                     offroad_mask,
                     routing_mask),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return ((transformed_img_feature,
                 torch.from_numpy(hist_points).float(),
                 torch.from_numpy(hist_points_step).float(),
                 coordinate_heading + np.pi / 2),
                (torch.from_numpy(pred_points).float(),
                 pred_boxs,
                 torch.empty(1),
                 pred_obs,
                 offroad_mask,
                 routing_mask,
                 torch.tensor([is_synthesized]),
                 torch.tensor([is_turning])))


class TrajectoryImitationCNNLSTMWithEnvLossDataset(Dataset):
    def __init__(self, data_dir, regions_list, renderer_config_file,
                 renderer_base_map_img_dir,
                 renderer_base_map_data_dir,
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

        self.img_bitmap_transform = transforms.Compose([
            # Normalized to [0, 1]
            transforms.ToTensor()])

        logging.info('Processing directory: {}'.format(data_dir))
        self.instances = file_utils.list_files(data_dir)

        self.total_num_data_pt = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_data_pt))

        self.chauffeur_net_feature_generator = \
            ChauffeurNetFeatureGenerator(regions_list,
                                         renderer_config_file,
                                         renderer_base_map_img_dir,
                                         renderer_base_map_data_dir)
        self.img_feature_rotation = img_feature_rotation
        self.past_motion_dropout = past_motion_dropout
        renderer_config = proto_utils.get_pb_from_text_file(
            renderer_config_file,
            planning_semantic_map_config_pb2.PlanningSemanticMapConfig())
        self.img_size = [renderer_config.width, renderer_config.height]
        self.max_rand_coordinate_heading = np.radians(
            renderer_config.max_rand_delta_phi)
        self.ouput_point_num = ouput_point_num
        self.evaluate_mode = evaluate_mode

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        frame_name = self.instances[idx]

        is_synthesized = 'is_synthesized' in frame_name
        is_turning = 'NO_TURN' not in frame_name

        frame = proto_utils.get_pb_from_bin_file(
            frame_name, learning_data_pb2.LearningDataFrame())

        region = frame.map_name

        coordinate_heading = 0.
        if self.img_feature_rotation:
            coordinate_heading = np.random.uniform() * 2 * self.max_rand_coordinate_heading - \
                self.max_rand_coordinate_heading

        is_past_motion_dropout = False
        if self.past_motion_dropout:
            is_past_motion_dropout = torch.rand(1) > 0.5

        # use adc_trajectory_point rather than localization
        # because of the use of synthesizing sometimes
        current_traj_point = frame.adc_trajectory_point[-1].trajectory_point
        current_path_point = current_traj_point.path_point
        current_x = current_path_point.x
        current_y = current_path_point.y
        current_theta = current_path_point.theta
        current_v = torch.tensor([current_traj_point.v]).float()

        img_feature = self.chauffeur_net_feature_generator.\
            render_stacked_img_features(region,
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
            heading_diff = NormalizeAngle(pred_theta - ref_coords[2])
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

        offroad_mask = self.chauffeur_net_feature_generator.\
            render_offroad_mask(region,
                                current_x,
                                current_y,
                                current_theta,
                                coordinate_heading)
        offroad_mask = self.img_bitmap_transform(offroad_mask)
        offroad_mask = offroad_mask.repeat(self.ouput_point_num, 1, 1, 1)

        routing_mask = self.chauffeur_net_feature_generator.\
            render_routing_mask(region,
                                current_x,
                                current_y,
                                current_theta,
                                frame.routing.local_routing_lane_id,
                                coordinate_heading)
        routing_mask = self.img_bitmap_transform(routing_mask)
        routing_mask = routing_mask.repeat(self.ouput_point_num, 1, 1, 1)

        if self.evaluate_mode:
            merged_img_feature = self.chauffeur_net_feature_generator.render_merged_img_feature(
                img_feature)
            return ((transformed_img_feature,
                     current_v),
                    (torch.from_numpy(pred_points).float(),
                     pred_boxs,
                     torch.empty(1),
                     pred_obs,
                     offroad_mask,
                     routing_mask),
                    merged_img_feature,
                    coordinate_heading,
                    frame.message_timestamp_sec)

        return ((transformed_img_feature,
                 current_v,
                 coordinate_heading + np.pi / 2),
                (torch.from_numpy(pred_points).float(),
                 pred_boxs,
                 torch.empty(1),
                 pred_obs,
                 offroad_mask,
                 routing_mask,
                 torch.tensor([is_synthesized]),
                 torch.tensor([is_turning])))
