#!/usr/bin/env python

import argparse

from learning_algorithms.prediction.data_preprocessing.features_labels_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge all label_dicts in each'
                                                 'terminal folder.')
    parser.add_argument('dirpath', type=str, help='Path of terminal folder.')
    args = parser.parse_args()

    MergeDicts(args.dirpath)
