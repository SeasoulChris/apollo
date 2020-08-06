#!/usr/bin/env python
from math import ceil
from random import shuffle
import os
import pickle

import fueling.common.logging as logging
from fueling.common.base_pipeline import BasePipeline
from fueling.common.file_utils import makedirs


def list_category(src_dir):
    return [f.name for f in os.scandir(src_dir) if f.is_dir()]


def get_training_and_testing_sets(elem, dst_dir, split_ratio=0.8):
    # pick split_ratio precent files
    cur_category, file_list = elem
    shuffle(file_list)
    split_index = ceil(len(file_list) * split_ratio)
    train_list_file = os.path.join(dst_dir, 'train', f'{cur_category}.txt')
    test_list_file = os.path.join(dst_dir, 'test', f'{cur_category}.txt')
    logging.info(f'train f{train_list_file} list length {len(file_list[:split_index])}')
    logging.info(f'test f{test_list_file} list length {len(file_list[split_index:])}')
    dump_file_list(train_list_file, file_list[:split_index])
    dump_file_list(test_list_file, file_list[split_index:])
    return (f'train f{train_list_file} list length {len(file_list[:split_index])}',
            f'test f{test_list_file} list length {len(file_list[split_index:])}')


def dump_file_list(dst_file, file_list):
    """ dump file list to dst_file """
    with open(dst_file, "wb") as fp:
        pickle.dump(file_list, fp)


class DataSplitter(BasePipeline):

    def run(self):
        object_storage = self.partner_storage() or self.our_storage()
        input_data_path = self.FLAGS.get('input_data_path')
        abs_source_dir = object_storage.abs_path(input_data_path)
        target_dir = object_storage.abs_path(self.FLAGS.get('output_data_path'))

        makedirs(os.path.join(target_dir, 'train'))
        makedirs(os.path.join(target_dir, 'test'))

        category_rdd = (self.to_rdd([abs_source_dir])
                        # Paired RDD (category)
                        .flatMap(list_category))
        logging.info(f'category numbers are {category_rdd.count()}')
        logging.info(f'category folder is {category_rdd.first()}')
        file_list_rdd = (
            category_rdd
            # RDD (category, category)
            .keyBy(lambda category: category)
            # Paired RDD (category, h5 file list)
            .mapValues(lambda category: object_storage.list_files(
                os.path.join(input_data_path, category), suffix='.h5'))
            # RDD (1)
            .map(lambda elem: get_training_and_testing_sets(elem, target_dir)))
        logging.info(f'file list numbers are {file_list_rdd.count()}')
        logging.info(f'file list first element is {file_list_rdd.first()}')


if __name__ == '__main__':
    DataSplitter().main()
