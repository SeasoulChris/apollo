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


class LearningDataOrganizer:
    '''
    An orginizer which takes in LearningData, outputs LearningDataFrame and 
        distribute them into training_set, validation_set and testing_set
    '''

    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.frames_total_count = None

        self.training_set_ratio = 0.8
        self.validation_set_ratio = 0.1
        self.testing_set_ratio = 0.1

        self._organize()

    def _organize(self):
        # Make output_dir
        if os.path.isdir(self.output_dir):
            logging.info(self.output_dir +
                         " directory exists, delete it!")
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)
        logging.info("Making output directory: " + self.output_dir)

        # List LearningData files and copy into LearningDataFrame .bin
        frames_list = []
        learning_data_file_paths = file_utils.list_files(self.data_dir)
        for file_path in learning_data_file_paths:
            if 'future_status' not in file_path or 'bin' not in file_path:
                continue
            learning_data_frames = proto_utils.get_pb_from_bin_file(
                file_path, learning_data_pb2.LearningData())
            frames_base_name = os.path.basename(file_path)
            for frame_num, learning_data_frame in enumerate(learning_data_frames.learning_data):
                frame_name = os.path.join(
                    self.output_dir, frames_base_name + "_{}.bin".format(frame_num))
                proto_utils.write_pb_to_bin_file(
                    learning_data_frame, frame_name)
                frames_list.append(frame_name)

        # Shuffle the frames_list and categorize frame into sub folders
        random.seed(0)
        random.shuffle(frames_list)
        self.frames_total_count = len(frames_list)
        training_set_end_idx = int(
            self.frames_total_count * self.training_set_ratio)
        validation_set_end_idx = training_set_end_idx + \
            int(self.frames_total_count * self.validation_set_ratio)
        if validation_set_end_idx == 0:
            logging.info(
                'frames num too small, no frame is distributed into validation set')
        training_set_list = frames_list[:training_set_end_idx]
        validation_set_list = frames_list[training_set_end_idx:validation_set_end_idx]
        testing_set_list = frames_list[validation_set_end_idx:]

        set_lists = [training_set_list, validation_set_list, testing_set_list]

        set_dirs = [os.path.join(self.output_dir, 'training_set/'),
                    os.path.join(self.output_dir, 'validation_set/'),
                    os.path.join(self.output_dir, 'testing_set/')]
        for set_dir in set_dirs:
            if os.path.isdir(set_dir):
                logging.info(set_dir +
                             " directory exists, delete it!")
                shutil.rmtree(set_dir)
            os.mkdir(set_dir)
            logging.info("Making output sub folders: " + set_dir)

        for set_idx, set_list in enumerate(set_lists):
            for frame_file in tqdm(set_list):
                shutil.move(frame_file, set_dirs[set_idx])

        train_set_files = file_utils.list_files(set_dirs[0])
        validation_set_files = file_utils.list_files(set_dirs[1])
        testing_set_files = file_utils.list_files(set_dirs[2])

        with open(os.path.join(self.output_dir, "frames_distribution_details.txt"), "w") \
                as output_file:
            output_file.write("train_set_has {} frames \n".format(len(train_set_files)))
            output_file.write("validation_set_has {} frames \n".format(len(validation_set_files)))
            output_file.write("test_set_has {} frames \n".format(len(testing_set_files)))

        return train_set_files, validation_set_files, testing_set_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='learning_data organizer data folder')
    parser.add_argument('data_dir', type=str,
                        help='organizer data folder')
    parser.add_argument('output_dir', type=str,
                        help='output data folder')
    args = parser.parse_args()

    learning_data_organizer = LearningDataOrganizer(args.data_dir, args.output_dir)
