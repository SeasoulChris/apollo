#!/usr/bin/env python

import argparse


if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(
        description='semantic_map model training pipeline')

    parser.add_argument('--train_dir', type=str, help='training data directory')
    parser.add_argument('--valid_dir', type=str, help='validation data directory')

    parser.add_argument('-s', '--save-path', type=str, default='./',
                        help='Specify the directory to save trained models.')

    args = parser.parse_args()
