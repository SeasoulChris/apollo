#!/usr/bin/env python

import argparse
import os
import pickle

from fueling.control.dynamic_model.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model.gp_regression.evaluation import test_gp
from fueling.control.dynamic_model.gp_regression.gaussian_process import GaussianProcess
from fueling.control.dynamic_model.gp_regression.train import train_gp


def launch(args):
    # tasks
    args.train_gp = False
    args.test_gp = True

    dataset = GPDataSet(args)
    if args.train_gp:
        # train Gaussian process model
        train_gp(args, dataset, GaussianProcess)
    if args.test_gp:
        # train Gaussian process model
        test_gp(args, dataset, GaussianProcess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument('--training_data_path', type=str,
                        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/dataset/training")
    parser.add_argument('--testing_data_path', type=str,
                        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/dataset/testing")
    parser.add_argument('--gp_model_path', type=str,
                        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/gp_model")
    parser.add_argument('--eval_result_path', type=str,
                        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/results")

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=10)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    args = parser.parse_args()
    launch(args)
