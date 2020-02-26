#!/usr/bin/env python

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import fueling.common.logging as logging

from fueling.common.coord_utils import CoordUtils
from modules.planning.proto.learning_data_pb2 import LearningData
import fueling.common.file_utils as file_utils
from learning_algorithms.prediction.data_preprocessing.map_feature.online_mapping import ObstacleMapping

LABEL_TRAJECTORY_POINT_NUM = 20
MAP_IMG_DIR = "/fuel/learning_algorithms/prediction/data_preprocessing/map_feature/"
ENABLE_IMG_DUMP = False


def LoadInstances(filepath):
    instances = LearningData()
    try:
        with open(filepath, 'rb') as file_in:
            instances.ParseFromString(file_in.read())
        return instances
    except BaseException:
        return None


class DataPreprocessor(object):
    def __init__(self, pred_len=3.0):
        self.pred_len = pred_len

    def load_numpy_dict(self, feature_file_path,
                        label_dir='labels',
                        label_file='instances_labels.npy'):
        '''Load the numpy dictionary file for the corresponding feature-file.
        '''
        dir_path = os.path.dirname(feature_file_path)
        label_path = os.path.join(dir_path, label_file)
        if not os.path.isfile(label_path):
            return None
        return np.load(label_path).item()

    def preprocess_data(self, instance_dir):
        '''
        params:
            - instance_dir: the directory containing all data_for_learn
        '''
        # Go through all the data_for_learning file, for each data-point, find
        # the corresponding label file, merge them.
        all_file_paths = file_utils.list_files(instance_dir)
        total_num_data_points = 0
        total_usable_data_points = 0

        file_count = 0
        for path in all_file_paths:
            file_count += 1
            print('============================================')
            print('Reading file: {}. ({}/{})'.format(path, file_count, len(all_file_paths)))

            # Load the feature for learning file.
            instances = LoadInstances(path)
            if instances is None:
                print('Failed to read instances file: {}.'.format(path))
                continue

            # Go through the entries in this feature file.
            total_num_data_points += len(instances.learning_data)
            output_np_array = []
            for instance in instances.learning_data:
                current_features = []
                current_label = []

                localization_feature = instance.localization
                current_features += [localization_feature.position.x]
                current_features += [localization_feature.position.y]
                current_features += [localization_feature.position.z]

                current_features += [localization_feature.heading]
                current_features += [localization_feature.linear_velocity.x]
                current_features += [localization_feature.linear_velocity.y]
                current_features += [localization_feature.linear_velocity.z]

                current_features += [localization_feature.linear_acceleration.x]
                current_features += [localization_feature.linear_acceleration.y]
                current_features += [localization_feature.linear_acceleration.z]

                chassis_feature = instance.chassis
                current_features += [chassis_feature.speed_mps]
                current_features += [chassis_feature.throttle_percentage]
                current_features += [chassis_feature.brake_percentage]
                current_features += [chassis_feature.steering_percentage]

                label_points = instance.trajectory_point
                # TODO(all): how to make sure all label points have the same number
                for trajectory_point in label_points:
                    current_label += [trajectory_point.path_point.x]
                    current_label += [trajectory_point.path_point.y]

                # if (LABEL_TRAJECTORY_POINT_NUM != len(current_label)):
                #     logging.warn("label point number:{}".format(len(current_label)))
                #     continue
                current_data_point = [current_features, current_label]
                # Update into the output_np_array.
                output_np_array.append(current_data_point)

            num_usable_data_points = len(output_np_array)

            # Save into a local file for training.
            try:
                logging.info('Total usable data points: {} out of {}.'.format(
                    num_usable_data_points, len(instances.learning_data)))
                output_np_array = np.array(output_np_array)
                np.save(path + '.training_data.npy', output_np_array)
                total_usable_data_points += num_usable_data_points
            except BaseException:
                logging.error('Failed to save output file:{}'.format(
                    path + '.training_data.npy'))

        logging.info('There are {} usable data points out of {}.'.format(
            total_usable_data_points, total_num_data_points))


class SemanticMapDataset(Dataset):
    def __init__(self, data_dir):
        self.map_region = []
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.instances = []

        self.reference_world_coord = []

        accumulated_data_pt = 0

        # TODO(Hongyi): add the drawing class here.
        self.base_map = {
            "sunnyvale": cv.imread(MAP_IMG_DIR + "sunnyvale_with_two_offices.png"),
            "san_mateo": cv.imread(MAP_IMG_DIR + "san_mateo.png")}

        logging.info('Processing directory: {}'.format(data_dir))
        all_file_paths = file_utils.list_files(data_dir)
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            logging.info("loading {} ...".format(file_path))
            file_content = np.load(file_path, allow_pickle=True).tolist()
            self.instances += file_content

        self.total_num_data_pt = len(self.instances)
        logging.info('Total number of data points = {}'.format(self.total_num_data_pt))

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        region = 'sunnyvale'
        world_coord = self.instances[idx][0][0:2]
        world_coord += [self.instances[idx][0][3]]  # heading
        # logging.info('world coord:{}'.format(world_coord))
        adc_mapping = ObstacleMapping(region, self.base_map[region], world_coord, None)

        adc_pose = [world_coord[0], world_coord[1]]
        img = adc_mapping.crop_by_rectangle(adc_pose)

        if self.img_transform:
            img = self.img_transform(img)

        if ENABLE_IMG_DUMP:
            cv.imwrite("/fuel/data/tmp/img{}.png".format(idx), img)

        # print("features:")
        # print(self.instances[idx][0])

        # print("label:")

        # print(self.instances[idx][1])
        # return ((img, self.instances[idx][0]),
        #          self.instances[idx][1])
        return ((img,
                torch.from_numpy(np.asarray(self.instances[idx][0])).float()),
                torch.from_numpy(np.asarray(self.instances[idx][1])).float())


if __name__ == '__main__':
    # Given cleaned labels, preprocess the data-for-learning and generate
    # training-data ready for torch Dataset.

    # bin file => numpy file
    # data_preprocessor = DataPreprocessor()
    # data_preprocessor.preprocess_data(
    #     '/apollo/modules/planning/data/instances/')

    # dump one instance image for debug
    dataset = SemanticMapDataset('/fuel/fueling/planning/datasets/training')
    # dataset[0]
    dataset[100]
