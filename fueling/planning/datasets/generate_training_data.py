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
    instances = LearningData()
    try:
        with open(filepath, 'rb') as file_in:
            instances.ParseFromString(file_in.read())
        return instances
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

            file_utils.makedirs(os.path.dirname(dest_dir))

            delete_filelist = glob.glob(dest_dir + '/' + file_name + '.training_data.npy')
            for delete_file in delete_filelist:
                if (os.path.exists(delete_file)):
                    os.remove(delete_file)

            file_count += 1
            logging.info('Reading file: {}. ({}/{})'.format(file_path,
                                                            file_count, len(all_file_paths)))

            # Load the feature for learning file.
            instances = LoadInstances(file_path)
            if instances is None:
                print('Failed to read instances file: {}.'.format(file_path))
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

                adc_trajectory_point = instance.adc_trajectory_point
                # TODO(all): how to make sure all label points have the same number
                for adc_trajectory_point in adc_trajectory_point:
                    current_label += [adc_trajectory_point.timestamp_sec]
                    current_label += [adc_trajectory_point.trajectory_point.path_point.x]
                    current_label += [adc_trajectory_point.trajectory_point.path_point.y]

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
