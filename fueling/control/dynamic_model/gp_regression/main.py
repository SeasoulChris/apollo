#!/usr/bin/env python

import argparse
import os
import pickle

from dataset import GPDataSet
from gaussian_process import GaussianProcess
from train import train_gp


def launch(args):
    # tasks
    args.train_gp = True
    args.post_tests = False

    dataset = GPDataSet(args)
    # train propagation Gaussian process
    train_gp(args, dataset, GaussianProcess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument('--labeled_data_path', type=str,
                        default="testdata/control/gaussian_process/dataset/label_generation/")
    parser.add_argument('--result_path', type=str,
                        default="testdata/control/gaussian_process/results/")

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=10)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    args = parser.parse_args()
    launch(args)
