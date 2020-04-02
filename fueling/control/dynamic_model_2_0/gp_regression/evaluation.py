#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
import fueling.common.logging as logging


def evaluation(args, dataset, GaussianProcess):
    """Load GP model params from files and make predictions and testset"""
    # get training data
    input_data, gt_data = dataset.get_test_data()
    # reformulate training data
    input_data = torch.transpose(input_data, 0, 1)
    input_dim = input_data.shape[-1]
    batch_size = input_data.shape[-2]
    output_dim = gt_data.shape[-1]

    logging.info(f'input data shape is {input_data.shape}')
    logging.info(f'output data shape is {gt_data.shape}')

    inducing_points = input_data[:, torch.arange(0, batch_size,
                                                 step=int(max(batch_size / args.num_inducing_point, 1))).long(), :]
    logging.info(f'inducing_points shape is {inducing_points.shape}')
    # load state dict
    file_path = os.path.join(args.gp_model_path, 'gp.pth')
    logging.info("************Loading GP model from {}".format(file_path))
    state_dict = torch.load(file_path)
    # model
    encoder_net_model = Encoder(u_dim=input_dim, kernel_dim=args.kernel_dim)
    gp_model = GPModel(inducing_points, encoder_net_model, output_dim)
    gp_model.load_state_dict(state_dict)
    # predicted results
    gp_model.eval()
    logging.info(input_data.shape)

    predictions = gp_model(input_data)
    mean = predictions.mean
    variance = predictions.variance
    logging.info(f'mean data shape is {mean.shape}')
    logging.info(f'variance shape is {variance.shape}')

    input_data = input_data.numpy()
    # Input feature visualization
    for i in range(len(input_data)):
        if i % 100 == 0:
            plt.figure(figsize=(4, 3))
            plt.title("Dataset Visualization")
            plt.plot(input_data[i, :, 0], 'o', color='black')
            plt.plot(input_data[i, :, 1], 'o', color='grey')
            plt.plot(input_data[i, :, 2], 'o', color='green')
            plt.plot(input_data[i, :, 3], 'o', color='red')
            plt.plot(input_data[i, :, 4], 'o', color='blue')
            plt.plot(input_data[i, :, 5], 'o', color='purple')
            plt.show()
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(gt_data, mean))
    logging.info("RMSE:{}".format(loss))
    gt_data = gt_data.numpy()
    mean = mean.detach().numpy()
    plt.figure(figsize=(4, 3))
    plt.title("Result Visualization")
    plt.plot(gt_data[:, 0], gt_data[:, 1], 'o', color='blue')
    plt.plot(mean[:, 0], mean[:, 1], 'o', color='red')
    plt.plot([gt_data[:, 0], mean[:, 0]], [gt_data[:, 1],
                                           mean[:, 1]], 'g:')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument(
        '--training_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/training")
    # default = "/fuel/fueling/control/dynamic_model_2_0/testdata/labeled_data"
    parser.add_argument(
        '--testing_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/test_dataset")
    parser.add_argument(
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model_output")
    parser.add_argument(
        '--eval_result_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/evaluation_results")
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
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    # argment to train or test gp
    parser.add_argument('--train_gp', type=bool, default=True)
    parser.add_argument('--test_gp', type=bool, default=True)

    args = parser.parse_args()
    dataset = GPDataSet(args)
    evaluation(args, dataset, GPModel)
