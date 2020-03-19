#!/usr/bin/env python

import argparse


from fueling.control.dynamic_model_2_0.evaluation.evaluation import test_gp
from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.gaussian_process import GaussianProcess
from fueling.control.dynamic_model_2_0.gp_regression.train import train_gp


def launch(args):

    dataset = GPDataSet(args)
    if args.train_gp:
        # train Gaussian process model
        train_gp(args, dataset, GaussianProcess)
    if args.test_gp:
        # test Gaussian process model
        test_gp(args, dataset, GaussianProcess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument(
        '--training_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/training")
    # default = "/fuel/fueling/control/dynamic_model_2_0/testdata/labeled_data")
    parser.add_argument(
        '--testing_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/test_dataset")
    parser.add_argument(
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model")
    parser.add_argument(
        '--eval_result_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/results")
    # parser.add_argument(
    #     '--online_gp_model_path',
    #     type=str,
    #     default="/fuel/fueling/control/dynamic_model_2_0/testdata/20191004-130454")

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=10)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    # argment to train or test gp
    parser.add_argument('--train_gp', type=bool, default=True)
    parser.add_argument('--test_gp', type=bool, default=True)

    args = parser.parse_args()
    launch(args)
