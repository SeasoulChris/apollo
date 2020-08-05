#!/usr/bin/env python
import argparse
import os

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import gpytorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
import fueling.common.logging as logging


def evaluation(args, dataset, GaussianProcess, data_frame_length=100, is_plot=False):
    """Load GP model params from files and make predictions and testset"""
    # get training data
    input_data, gt_data = dataset.get_test_data()
    features, labels = dataset.get_train_data()
    # reformulate training data
    input_data = torch.transpose(input_data, 0, 1)

    features = torch.transpose(features, 0, 1)
    # [batch_size, channel] (window_size = 1)
    labels = torch.transpose(labels, 0, 1)

    # input_dim = input_data.shape[-1]
    # batch_size = input_data.shape[-2]
    output_dim = gt_data.shape[-1]

    logging.debug(f'input data shape is {input_data.shape}')
    logging.info(f'output data shape is {gt_data.shape}')
    logging.info("************Input Dim: {}".format(features.shape))
    logging.info("************Output Dim: {}".format(labels.shape))
    step_size = int(max(features.shape[-2] / args.num_inducing_point, 1))
    logging.info(f'step_size: {step_size}')

    inducing_points = features[:, torch.arange(0, features.shape[-2], step=step_size), :]

    logging.info(f'inducing_points shape is {inducing_points.shape}')
    # load state dict
    file_path = os.path.join(args.gp_model_path, 'gp.pth')
    # file_path = glob.glob(os.path.join(args.gp_model_path, '*.pth'))
    logging.info("************Loading GP model from {}".format(file_path))
    model_state_dict, likelihood_state_dict = torch.load(file_path)
    # model
    encoder_net_model = Encoder(u_dim=features.shape[-1], kernel_dim=args.kernel_dim)
    # TODO(Shu): check if it is necessary to use training data for initialization
    gp_model = GPModel(inducing_points, encoder_net_model, output_dim)
    gp_model.load_state_dict(model_state_dict)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)
    likelihood.load_state_dict(likelihood_state_dict)
    # predicted results
    gp_model.eval()
    likelihood.eval()
    logging.info(input_data.shape)

    predictions = likelihood(gp_model(input_data))
    lower, upper = predictions.confidence_region()
    mean = predictions.mean
    variance = predictions.variance
    logging.info(f'mean data shape is {mean.shape}')
    logging.info(f'variance shape is {variance.shape}')
    if is_plot:
        input_data = input_data.numpy()
        # Input feature visualization
        for i in range(len(input_data)):
            if i % data_frame_length == 0:
                plt.figure(figsize=(4, 3))
                plt.title("Dataset Visualization")
                plt.plot(input_data[:, i, 0], 'o', color='black')
                plt.plot(input_data[:, i, 1], 'o', color='grey')
                plt.plot(input_data[:, i, 2], 'o', color='green')
                plt.plot(input_data[:, i, 3], 'o', color='red')
                plt.plot(input_data[:, i, 4], 'o', color='blue')
                plt.plot(input_data[:, i, 5], 'o', color='purple')
                plt.show()
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(gt_data, mean))
        logging.info("RMSE:{}".format(loss))
        gt_data = gt_data.numpy()
        mean = mean.detach().numpy()
        upper = upper.detach().numpy()
        lower = lower.detach().numpy()
        plt.figure(figsize=(4, 3))
        plt.title("Result Visualization")
        plt.plot(gt_data[:, 0], gt_data[:, 1], 'o', color='blue')
        plt.plot(mean[:, 0], mean[:, 1], 'o', color='red')
        plt.plot([gt_data[:, 0], mean[:, 0]], [gt_data[:, 1], mean[:, 1]], 'g:')
        plt.show()
    # save as npy, avoid regenerate result when modifying plots.
    evaluation_result = dict()
    evaluation_result['validation_labels'] = gt_data.numpy()
    evaluation_result['training_labels'] = labels.numpy()
    logging.info(labels.numpy())
    evaluation_result['mean'] = mean.detach().numpy()
    evaluation_result['upper'] = upper.detach().numpy()
    evaluation_result['lower'] = lower.detach().numpy()
    np.save(os.path.join(args.testing_data_path, 'evaluation_result.npy'), evaluation_result)
    logging.info(f'upper shape: {upper.shape}')
    return mean.detach().numpy()


def evaluation_visualization(file_path):
    evaluation_result = np.load(file_path, allow_pickle=True).item()
    validation_labels = evaluation_result['validation_labels']
    mean = evaluation_result['mean']
    logging.info(mean)
    upper = evaluation_result['upper']
    lower = evaluation_result['lower']
    training_labels = evaluation_result['training_labels']
    # plot
    fig, ax = plt.subplots(1)
    ax.set_xlabel('$\\Delta$x (m)', fontdict={'size': 12})
    ax.set_ylabel('$\\Delta$y (m)', fontdict={'size': 12})
    ax.set_title("Result Visualization")
    # confidence region
    confidence_regions = []
    for idx in range(0, validation_labels.shape[0]):
        rect = Rectangle((lower[idx, 0], lower[idx, 1]), upper[idx, 0]
                         - lower[idx, 0], upper[idx, 1] - lower[idx, 1])
        confidence_regions.append(rect)
    pc = PatchCollection(confidence_regions, facecolor='g', alpha=0.04, edgecolor='b')
    ax.add_collection(pc)
    ax.plot(validation_labels[:, 0], validation_labels[:, 1],
            'o', color='blue', label='Ground truth')
    ax.plot(training_labels[:, 0], training_labels[:, 1],
            'kx', label='Training ground truth')
    ax.plot(mean[:, 0], mean[:, 1], 's', color='r', label='Predicted mean')
    ax.legend(fontsize=12, frameon=False)
    ax.grid(True)
    # timestamp
    plt.savefig(os.path.join(os.path.dirname(file_path), "plot.png"))
    plt.show()


def plot(validation_labels, mean, upper, lower):
    # 3D plot delta_x, delta_y for each training point
    fig, ax = plt.subplot(1)
    plt.xlabel('$\\delta$x (m)', fontdict={'size': 12})
    plt.ylabel('$\\delta$y (m)', fontdict={'size': 12})
    plt.title("Result Visualization")
    plt.plot(validation_labels[:, 0], validation_labels[:, 1],
             'o', color='blue', label='Ground truth')
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True)
    plt.plot(mean[:, 0], mean[:, 1], 's', color='b', label='Predicted mean')
    # confidence region
    confidence_region = []
    rect = Rectangle((lower[0], lower[1]), upper[0] - lower[0], upper[1] - lower[1])
    confidence_region.append(rect)
    pc = PatchCollection(confidence_region, facecolor='b', alpha=0.005, edgecolor='b')
    ax.add_collection(pc)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument('-train',
                        '--training_data_path',
                        type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/testdata/0417/train")
    parser.add_argument(
        '-test',
        '--testing_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/0417/validation")
    parser.add_argument(
        '-md',
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model_output/20200420-214841")
    parser.add_argument(
        '--eval_result_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/evaluation_results")

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=128)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    # argment to train or test gp
    parser.add_argument('--train_gp', type=bool, default=True)
    parser.add_argument('--test_gp', type=bool, default=True)

    # argment to use cuda or not
    parser.add_argument('--use_cuda', type=bool, default=False)
    args = parser.parse_args()

    result_file = os.path.join(args.testing_data_path + 'evaluation_result.npy')
    if not os.path.exists(result_file):
        dataset = GPDataSet(args)
        evaluation(args, dataset, GPModel)
    evaluation_visualization(result_file)
