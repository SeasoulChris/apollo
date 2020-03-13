#!/usr/bin/env python
import glob
import numpy as np
import os
import re
import time

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
from modules.planning.proto.learning_data_pb2 import LearningData

LABEL_TRAJECTORY_POINT_NUM = 20


def LoadInstances(filepath):
    learning_data = LearningData()
    try:
        with open(filepath, 'rb') as file_in:
            learning_data.ParseFromString(file_in.read())
        return learning_data
    except BaseException:
        return None


class GenerateTrainingData(BasePipeline):
    """Learning data to training data"""

    def __init__(self):
        self.src_dir_prefix = 'modules/planning/learning_data'
        self.dest_dir_prefix = 'modules/planning/training_data'

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

    def run_test(self):
        """Run Test"""
        self.src_dir_prefix = '/apollo/data/learning_data'
        self.dest_dir_prefix = '/apollo/data/training_data'

        src_dirs_set = set([])
        for root, dirs, files in os.walk(self.src_dir_prefix):
            for file in files:
                src_dirs_set.add(root)

        processed_records = self.to_rdd(src_dirs_set).map(self.process_learning_data)

        logging.info('Processed {}/{} records'.format(processed_records.count(),
                                                      len(src_dirs_set)))
        return 0

    def run(self):
        """Run"""
        records_rdd = (self.to_rdd(self.our_storage().list_files(self.src_dir_prefix))
                       .map(os.path.dirname)
                       .distinct())

        processed_records = records_rdd.map(self.process_learning_data)

        logging.info('Processed {} records'.format(processed_records.count()))

    def add_chassis_feature(self, learning_data_frame, current_features):
        chassis_feature = learning_data_frame.chassis

        current_features.append(chassis_feature.speed_mps)
        current_features.append(chassis_feature.throttle_percentage)
        current_features.append(chassis_feature.brake_percentage)
        current_features.append(chassis_feature.steering_percentage)
        current_features.append(chassis_feature.gear_location)

        return current_features

    def add_localization_feature(self, learning_data_frame, current_features):
        localization_feature = learning_data_frame.localization

        current_features.append(localization_feature.position.x)
        current_features.append(localization_feature.position.y)
        current_features.append(localization_feature.position.z)

        current_features.append(localization_feature.heading)

        current_features.append(localization_feature.linear_velocity.x)
        current_features.append(localization_feature.linear_velocity.y)
        current_features.append(localization_feature.linear_velocity.z)

        current_features.append(localization_feature.linear_acceleration.x)
        current_features.append(localization_feature.linear_acceleration.y)
        current_features.append(localization_feature.linear_acceleration.z)

        current_features.append(localization_feature.angular_velocity.x)
        current_features.append(localization_feature.angular_velocity.y)
        current_features.append(localization_feature.angular_velocity.z)

        return current_features

    def add_obstacle_feature(self, learning_data_frame, current_features):
        obstacle_feature = learning_data_frame.obstacle

        # obstacle
        for obstacle in obstacle_feature:
            current_features.append(obstacle.id)
            current_features.append(obstacle.length)
            current_features.append(obstacle.width)
            current_features.append(obstacle.height)
            current_features.append(obstacle.type)

            # obstacle_trajectory_point
            for trajectory_point in obstacle.obstacle_trajectory_point:
                current_features.append(trajectory_point.position.x)
                current_features.append(trajectory_point.position.y)
                current_features.append(trajectory_point.position.z)
                current_features.append(trajectory_point.theta)
                current_features.append(trajectory_point.velocity.x)
                current_features.append(trajectory_point.velocity.y)
                current_features.append(trajectory_point.velocity.z)
                current_features.append(trajectory_point.acceleration.x)
                current_features.append(trajectory_point.acceleration.y)
                current_features.append(trajectory_point.acceleration.z)

                for point in trajectory_point.polygon_point:
                    current_features.append(point.x)
                    current_features.append(point.y)
                    current_features.append(point.z)

            # obstacle_prediction
            current_features.append(obstacle.obstacle_prediction.predicted_period)
            current_features.append(obstacle.obstacle_prediction.intent.type)
            current_features.append(obstacle.obstacle_prediction.priority.priority)
            current_features.append(obstacle.obstacle_prediction.is_static)

            for tj in obstacle.obstacle_prediction.trajectory:
                current_features.append(tj.probability)
                for point in tj.trajectory_point:
                    current_features.append(point.path_point.x)
                    current_features.append(point.path_point.y)
                    current_features.append(point.path_point.z)
                    current_features.append(point.path_point.theta)
                    current_features.append(point.path_point.s)
                    current_features.append(point.path_point.lane_id)

                    current_features.append(point.v)
                    current_features.append(point.a)
                    current_features.append(point.relative_time)
                    current_features.append(point.gaussian_info.sigma_x)
                    current_features.append(point.gaussian_info.sigma_y)
                    current_features.append(point.gaussian_info.correlation)
                    current_features.append(point.gaussian_info.area_probability)
                    current_features.append(point.gaussian_info.ellipse_a)
                    current_features.append(point.gaussian_info.ellipse_b)
                    current_features.append(point.gaussian_info.theta_a)
        return current_features

    def add_routing_response_feature(self, learning_data_frame, current_features):
        routing_response_feature = learning_data_frame.routing_response
        for lane_id in routing_response_feature.lane_id:
            current_features.append(lane_id)
        return current_features

    def add_traffic_light_feature(self, learning_data_frame, current_features):
        traffic_light_feature = learning_data_frame.traffic_light
        for traffic_light in traffic_light_feature:
            current_features.append(traffic_light.id)
            current_features.append(traffic_light.color)
        return current_features

    def generate_label(self, learning_data_frame, current_label):
        adc_trajectory_point = learning_data_frame.adc_trajectory_point
        # TODO(all): how to make sure all label points have the same number
        for adc_trajectory_point in adc_trajectory_point:
            current_label.append(adc_trajectory_point.timestamp_sec)
            current_label.append(adc_trajectory_point.trajectory_point.path_point.x)
            current_label.append(adc_trajectory_point.trajectory_point.path_point.y)
            current_label.append(adc_trajectory_point.trajectory_point.path_point.z)
            current_label.append(adc_trajectory_point.trajectory_point.v)
            current_label.append(adc_trajectory_point.trajectory_point.a)
            current_label.append(adc_trajectory_point.trajectory_point.relative_time)
            current_label.append(adc_trajectory_point.trajectory_point.da)
            current_label.append(adc_trajectory_point.trajectory_point.steer)
            current_label.append(adc_trajectory_point.trajectory_point.gaussian_info.sigma_x)
            current_label.append(adc_trajectory_point.trajectory_point.gaussian_info.sigma_y)
            current_label.append(adc_trajectory_point.trajectory_point.gaussian_info.correlation)
            current_label.append(adc_trajectory_point.trajectory_point.gaussian_info.area_probability)
            current_label.append(adc_trajectory_point.trajectory_point.gaussian_info.ellipse_a)
            current_label.append(adc_trajectory_point.trajectory_point.gaussian_info.ellipse_b)
            current_label.append(adc_trajectory_point.trajectory_point.gaussian_info.theta_a)
        return current_label

    def process_learning_data_frame(self, learning_data_frame):
        current_features = []
        current_label = []

        # chassis
        self.add_chassis_feature(learning_data_frame, current_features)

        # localization
        self.add_localization_feature(learning_data_frame, current_features)

        # obstacle
        self.add_obstacle_feature(learning_data_frame, current_features)

        # routing_response
        self.add_routing_response_feature(learning_data_frame, current_features)

        # traffic_light
        self.add_traffic_light_feature(learning_data_frame, current_features)

        # label
        self.generate_label(learning_data_frame, current_label)

        # if (LABEL_TRAJECTORY_POINT_NUM != len(current_label)):
        #     logging.warn("label point number:{}".format(len(current_label)))
        #     continue
        current_data_point = [current_features, current_label]
        return current_data_point

    def process_learning_data(self, src_dir):
        """ Process learning data: learning datas -> training data """
        # Go through all the learning_data file, for each data-point, find
        # the corresponding label file, merge them.
        all_file_paths = file_utils.list_files(src_dir)

        # sort by filenames numerically: learning_data.<int>.bin.training_data.npy
        all_file_paths.sort(
            key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        total_num_data_points = 0
        total_usable_data_points = 0

        file_count = 0
        for file_path in all_file_paths:
            if (not file_path.endswith('.bin')):
                continue

            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            src_dir_elements = file_dir.split("/")
            dest_dir_elements = ['training_data' if x ==
                                 'learning_data' else x for x in src_dir_elements]
            if ('training_data' in dest_dir_elements):
                dest_dir = "/".join(dest_dir_elements)
            else:
                dest_dir = "/".join(src_dir_elements)

            file_utils.makedirs(dest_dir)

            delete_filelist = glob.glob(dest_dir + '/' + file_name + '.training_data.npy')
            for delete_file in delete_filelist:
                if (os.path.exists(delete_file)):
                    os.remove(delete_file)

            file_count += 1
            logging.info('Reading file: {}. ({}/{})'.format(file_path,
                                                            file_count, len(all_file_paths)))

            # Load the feature for learning file.
            learning_data = LoadInstances(file_path)
            if learning_data is None:
                logging.info('Failed to read learning_data file: {}.'.format(file_path))
                continue

            # Go through the entries in this feature file.
            total_num_data_points += len(learning_data.learning_data)
            output_np_array = []
            for learning_data_frame in learning_data.learning_data:
                current_data_point = self.process_learning_data_frame(learning_data_frame)
                # Update into the output_np_array.
                output_np_array.append(current_data_point)

            num_usable_data_points = len(output_np_array)

            # Save into a local file for training.
            try:
                logging.info('Total usable data points: {} out of {}.'.format(
                    num_usable_data_points, len(learning_data.learning_data)))
                output_np_array = np.array(output_np_array)
                new_file_path = dest_dir + '/' + file_name + '.training_data.npy'
                np.save(new_file_path, output_np_array)
                total_usable_data_points += num_usable_data_points
            except BaseException:
                logging.error('Failed to save output file:{}'.format(
                    path + '.training_data.npy'))

        logging.info('There are {} usable data points out of {}.'.format(
            total_usable_data_points, total_num_data_points))


if __name__ == '__main__':
    GenerateTrainingData().main()
