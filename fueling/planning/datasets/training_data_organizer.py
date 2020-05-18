#!/usr/bin/env python

import argparse
import os
import random
import shutil
from tqdm import tqdm

from modules.planning.proto import learning_data_pb2

import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils


class TrainingDataOrganizer:
    '''
    An orginizer which takes in LearningData, outputs LearningDataFrame and 
        distribute them into training_set, validation_set and testing_set
    '''

    def __init__(self):
        self.training_set_ratio = 0.8
        self.validation_set_ratio = 0.1
        self.testing_set_ratio = 0.1
        self.to_be_synthesized_set_ratio = self.training_set_ratio * 0.1

        random.seed(0)

    def split_to_frames(self, data_dir, output_dir):
        # Make output_dir
        if os.path.isdir(output_dir):
            logging.info(output_dir +
                         " directory exists, delete it!")
            shutil.rmtree(output_dir)
        file_utils.makedirs(output_dir)
        logging.info("Making output directory: " + output_dir)

        # List LearningData files and copy into LearningDataFrame .bin
        frames_list = []
        learning_data_file_paths = file_utils.list_files(data_dir)
        logging.info("{} learning_data.bin to deal with in total".format(
            len(learning_data_file_paths)))
        for file_path in tqdm(learning_data_file_paths):
            if 'future_status' not in file_path or 'bin' not in file_path:
                continue
            learning_data_frames = proto_utils.get_pb_from_bin_file(
                file_path, learning_data_pb2.LearningData())
            frames_base_name = os.path.basename(file_path)
            for frame_num, learning_data_frame in enumerate(learning_data_frames.learning_data):
                frame_name = os.path.join(
                    output_dir, frames_base_name + ".frame_num_{}.bin".format(frame_num))
                proto_utils.write_pb_to_bin_file(
                    learning_data_frame, frame_name)
                frames_list.append(frame_name)

        logging.info('{} frames are split to dir {} from .bin files in dir {}'.format(
            len(frames_list), output_dir, data_dir))

        random.shuffle(frames_list)

        return frames_list

    def load_frames_list(self, frames_dir):
        frames_list = list(filter(lambda frame_path:
                           'future_status' in frame_path and 'bin' in frame_path,
                           file_utils.list_files(frames_dir)))
        
        random.shuffle(frames_list)
        
        return frames_list

    def pick_small_samples(self, samples_size, frames_list):

        if samples_size > len(frames_list):
            logging.info(
                'Desired samples size is larger than the size of given frames')

        return frames_list[:samples_size]

    def categorize_learning_sets(self, frames_list, output_dir):
        frames_total_count = len(frames_list)
        to_be_synthesized_set_end_idx = int(frames_total_count *
                                            self.to_be_synthesized_set_ratio)
        training_set_end_idx = int(
            frames_total_count * self.training_set_ratio)
        validation_set_end_idx = training_set_end_idx + \
            int(frames_total_count * self.validation_set_ratio)
        if validation_set_end_idx == 0:
            logging.info(
                'frames num too small, no frame is distributed into validation set')
        to_be_synthesized_set_list = frames_list[:to_be_synthesized_set_end_idx]
        training_set_list = frames_list[to_be_synthesized_set_end_idx:training_set_end_idx]
        validation_set_list = frames_list[training_set_end_idx:validation_set_end_idx]
        testing_set_list = frames_list[validation_set_end_idx:]

        set_lists = [to_be_synthesized_set_list,
                     training_set_list, validation_set_list, testing_set_list]

        set_dirs = [os.path.join(output_dir, 'to_be_synthesized_set/'),
                    os.path.join(output_dir, 'training_set/'),
                    os.path.join(output_dir, 'validation_set/'),
                    os.path.join(output_dir, 'testing_set/')]
        for set_dir in set_dirs:
            if os.path.isdir(set_dir):
                logging.info(set_dir +
                             " directory exists, delete it!")
                shutil.rmtree(set_dir)
            file_utils.makedirs(set_dir)
            logging.info("Making output sub folders: " + set_dir)

        for set_idx, set_list in enumerate(set_lists):
            for frame_file in tqdm(set_list):
                shutil.move(frame_file, set_dirs[set_idx])

        to_be_synthesized_set_files = file_utils.list_files(set_dirs[0])
        train_set_files = file_utils.list_files(set_dirs[1])
        validation_set_files = file_utils.list_files(set_dirs[2])
        testing_set_files = file_utils.list_files(set_dirs[3])

        with open(os.path.join(output_dir, "frames_distribution_details.txt"), "w") \
                as output_file:
            output_file.write(
                "train_set_has {} frames \n".format(len(train_set_files)))
            output_file.write("validation_set_has {} frames \n".format(
                len(validation_set_files)))
            output_file.write(
                "test_set_has {} frames \n".format(len(testing_set_files)))
            output_file.write(
                "to_besynthesized_set_has {} frames \n"
                .format(len(to_be_synthesized_set_files)))

        return train_set_files, validation_set_files, testing_set_files, \
            to_be_synthesized_set_files

    def run(self, data_dir, output_dir):
        total_frames_list = self.split_to_frames(data_dir, output_dir)
        return self.categorize_learning_sets(total_frames_list, output_dir)

    def run_on_frames(self, frames_dir, output_dir):
        total_frames_list = self.load_frames_list(frames_dir)
        return self.categorize_learning_sets(total_frames_list, output_dir)

    def run_on_small_samples(self, frames_dir, samples_size, output_dir):
        total_frames_list = self.load_frames_list(frames_dir)
        small_samples_list = self.pick_small_samples(samples_size, total_frames_list)
        return self.categorize_learning_sets(small_samples_list, output_dir)


if __name__ == "__main__":
    #TODO(Jinyun): use absl flag
    parser = argparse.ArgumentParser(
        description='learning_data organizer data folder')
    parser.add_argument('data_dir', type=str,
                        help='organizer data folder')
    parser.add_argument('output_dir', type=str,
                        help='output data folder')
    args = parser.parse_args()

    learning_data_organizer = TrainingDataOrganizer()

    learning_data_organizer.run(args.data_dir, args.output_dir)
    # learning_data_organizer.run_on_frames(args.output_dir, args.output_dir)
    # learning_data_organizer.run_on_small_samples(args.output_dir, 500000, args.output_dir)