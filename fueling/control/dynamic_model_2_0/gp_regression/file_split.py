#!/usr/bin/env python


from math import floor
from random import shuffle
import argparse
import os
import random
import shutil


import fueling.common.logging as logging
from fueling.common import file_utils


class TrainTestFileSplitter():
    def __init__(self, dst_dir):
        super().__init__()
        self.dst_dir = dst_dir
        self.category_dir_list = None
        self.split_ratio = 0.8

    def get_category_list(self):
        self.category_dir_list = [f.path for f in os.scandir(self.dst_dir) if f.is_dir()]
        logging.info(self.category_dir_list)
        # loop over each category
        for category_dir in self.category_dir_list:
            # get all files in all subfolders
            files = file_utils.list_files_with_suffix(category_dir, '.h5')
            # shuffle all the files
            shuffle(files)
            train_files, test_files = self.get_training_and_testing_sets(files)
            logging.info(f'training files are {len(train_files)}')
            logging.info(f'test files are {len(test_files)}')
            self.move_files(train_files, 'split', 'train')
            self.move_files(test_files, 'split', 'test')

    def get_training_and_testing_sets(self, file_list):
        # pick split_ratio precent files
        split_index = floor(len(file_list) * self.split_ratio)
        training = file_list[:split_index]
        testing = file_list[split_index:]
        return training, testing

    def move_files(self, files, src_prefix, dst_prefix):
        # move files to seperatored folder
        for file in files:
            dst_file = file.replace(src_prefix, dst_prefix, 1)
            dst_dir = os.path.dirname(dst_file)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.move(file, dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training validation data splitter')
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--dst_dir', type=str)
    args = parser.parse_args()
    splitter = TrainTestFileSplitter(args.src_dir)
    splitter.get_category_list()
