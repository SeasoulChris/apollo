#!/usr/bin/env python

import argparse

from fueling.prediction.learning.data_preprocessing.features_labels_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge all label_dicts in each'
                                                 'terminal folder.')
    parser.add_argument('features_dirpath', type=str,
                        help='Path of terminal folder for data_for_learn.')
    parser.add_argument('labels_dirpath', type=str,
                        help='Path of terminal folder for labels')
    args = parser.parse_args()

    list_of_files = os.listdir(args.features_dirpath)
    for file in list_of_files:
        full_file_path = os.path.join(args.features_dirpath, file)
        if file.split('.')[-1] == 'bin' and \
           file.split('.')[0] == 'datalearn':
            label_path = args.labels_dirpath
            CombineFeaturesAndLabels(full_file_path, label_path + '/labels.npy')

    MergeCombinedFeaturesAndLabels(args.features_dirpath)
