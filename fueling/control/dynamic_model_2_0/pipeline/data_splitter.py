#!/usr/bin/env python
from math import ceil
from random import shuffle
import os
import pickle

import fueling.common.logging as logging
from fueling.common.base_pipeline import BasePipeline
from fueling.common.file_utils import list_files_with_suffix, makedirs, touch


def list_category(src_dir):
    return [(f.name, f.path) for f in os.scandir(src_dir) if f.is_dir()]


def get_training_and_testing_sets(elem, dst_dir, split_ratio=0.8):
    # pick split_ratio precent files
    cur_category, file_list = elem
    shuffle(file_list)
    split_index = ceil(len(file_list) * split_ratio)
    train_list_file = os.path.join(dst_dir, 'train', f'{cur_category}.txt')
    test_list_file = os.path.join(dst_dir, 'test', f'{cur_category}.txt')
    touch(train_list_file)
    touch(test_list_file)
    print(f'train list {file_list[:split_index]}')
    print(f'test list {file_list[split_index:]}')
    dump_file_list(train_list_file, file_list[:split_index])
    dump_file_list(test_list_file, file_list[split_index:])
    return 1


def dump_file_list(dst_file, file_list):
    """ dump file list to dst_file """
    with open(dst_file, "wb") as fp:
        pickle.dump(file_list, fp)


class DataSplitter(BasePipeline):

    def run(self):
        object_storage = self.partner_storage() or self.our_storage()
        source_dir = object_storage.abs_path(self.FLAGS.get('input_data_path'))
        target_dir = object_storage.abs_path(self.FLAGS.get('output_data_path'))
        makedirs(os.path.join(target_dir, 'train'))
        makedirs(os.path.join(target_dir, 'test'))

        category_rdd = (self.to_rdd([source_dir])
                        # Paired RDD (category key, category dir)
                        .flatMap(list_category))
        logging.info(f'category numbers are {category_rdd.count()}')
        logging.info(f'category folder is {category_rdd.first()}')
        file_list_rdd = (
            category_rdd
            # Paired RDD (category key, h5 file lists in category)
            .mapValues(lambda category_dir: list_files_with_suffix(category_dir, '.h5'))
            .map(lambda elem: get_training_and_testing_sets(elem, target_dir)))
        logging.info(f'file list numbers are {file_list_rdd.count()}')
        logging.info(f'file list first element is {file_list_rdd.first()}')


if __name__ == '__main__':
    DataSplitter().main()
