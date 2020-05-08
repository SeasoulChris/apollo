#!/usr/bin/env python
import argparse
import glob
import os


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


from fueling.common import file_utils
from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset import DynamicModelDataset
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
import fueling.common.logging as logging


class ValidationVisualization():
    """ visualize validation results """

    def __init__(self, args):
        super().__init__()
        self.model = None
        self.likelihood = None
        self.inducing_points = None
        self.model_path = args.gp_model_path
        self.validation_data_path = args.validation_data_path
        self.inducing_point_file = os.path.join(self.validation_data_path, 'inducing_points.npy')
        self.dst_file_path = args.eval_result_path
        self.evaluation_result_file = None

        # parameters
        self.kernel_dim = args.kernel_dim
        self.input_dim = None
        self.output_dim = None

    def load_model(self):
        """
        load state dict for model and likelihood
        """
        # load state dict
        file_path = os.path.join(self.model_path, 'gp_model.pth')
        logging.info(f"Loading GP model from {file_path}")
        model_state_dict, likelihood_state_dict = torch.load(file_path)
        # encoder model
        encoder_net_model = Encoder(u_dim=self.input_dim, kernel_dim=self.kernel_dim)
        # TODO(Shu): check if it is necessary to use training data for initialization
        self.model = GPModel(self.inducing_points, encoder_net_model, self.output_dim)
        self.model.load_state_dict(model_state_dict)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.output_dim)
        self.likelihood.load_state_dict(likelihood_state_dict)

    def load_data(self, test_features, test_labels):
        """
        validation data and inducint points
        """
        # features: [num_of_data_points * window_size * input_channel]
        # labels: [same num_of_data_points * output_channel]
        self.input_dim = test_features.shape[-1]
        self.output_dim = test_labels.shape[-1]
        # get inducing points
        self.inducing_points = torch.from_numpy(
            np.load(self.inducing_point_file, allow_pickle=True))

    def evaluation(self, test_x, test_y, set_id, train_y=None):
        """
            result for each validation batch
        """
        self.model.eval()
        self.likelihood.eval()
        test_x = torch.transpose(test_x, 0, 1).type(torch.FloatTensor)
        logging.info(test_x.shape)
        # make prediction with input with uncertainty
        predictions = self.likelihood(self.model(test_x))
        lower, upper = predictions.confidence_region()
        mean = predictions.mean
        variance = predictions.variance
        logging.info(f'mean data shape is {mean.shape}')
        logging.info(f'variance shape is {variance.shape}')
        evaluation_result = dict()
        evaluation_result['validation_labels'] = test_y.numpy()
        if train_y.numpy().any():
            evaluation_result['training_labels'] = train_y.numpy()
        evaluation_result['mean'] = mean.detach().numpy()
        evaluation_result['upper'] = upper.detach().numpy()
        evaluation_result['lower'] = lower.detach().numpy()
        np.save(os.path.join(args.validation_data_path,
                             f'{set_id}_evaluation_result.npy'), evaluation_result)
        return mean.detach().numpy()

    def evaluation_visualization(self):
        # get data from .npy file
        logging.info(self.evaluation_result_file)
        evaluation_result = np.load(self.evaluation_result_file, allow_pickle=True).item()
        # ground truth
        validation_labels = evaluation_result['validation_labels']
        # predicted mean value
        mean = evaluation_result['mean']
        # confidence region
        upper = evaluation_result['upper']
        lower = evaluation_result['lower']
        # training data
        training_labels = evaluation_result['training_labels']

        # plot
        fig, ax = plt.subplots(1)
        ax.set_xlabel('$\Delta$x (m)', fontdict={'size': 12})
        ax.set_ylabel('$\Delta$y (m)', fontdict={'size': 12})
        ax.set_title("Result Visualization")
        # confidence region
        confidence_regions = []
        for idx in range(0, validation_labels.shape[0]):
            logging.info(f'rectangle number is {idx}')
            rect = Rectangle((lower[idx, 0], lower[idx, 1]), upper[idx, 0] -
                             lower[idx, 0], upper[idx, 1] - lower[idx, 1])
            confidence_regions.append(rect)
        # confident regions
        pc = PatchCollection(confidence_regions, facecolor='g', alpha=0.04, edgecolor='b')
        ax.add_collection(pc)
        # ground truth (dx, dy)
        ax.plot(validation_labels[:, 0], validation_labels[:, 1],
                'o', color='blue', label='Ground truth')
        # training labels
        ax.plot(training_labels[:, 0], training_labels[:, 1],
                'kx', label='Training ground truth')
        # predicted mean value
        ax.plot(mean[:, 0], mean[:, 1], 's', color='r', label='Predicted mean')
        ax.legend(fontsize=12, frameon=False)
        ax.grid(True)
        # time stamp
        file_utils.makedirs(self.dst_file_path)
        plt.savefig(os.path.join(self.dst_file_path, "plot.png"))
        plt.show()

    def validation(self, test_features, test_labels, set_id, train_labels):
        self.load_data(test_features, test_labels)
        logging.info(self.inducing_point_file)
        logging.info(self.inducing_points.shape)
        self.load_model()
        # # check if evaluation result exists
        logging.info(f'Validating set {set_id}')
        self.evaluation_result_file = os.path.join(args.validation_data_path,
                                                   f'{set_id}_evaluation_result.npy')
        if not os.path.exists(self.evaluation_result_file):
            self.evaluation(test_features, test_labels, set_id, train_labels)
            logging.info(f'Evaluation results are saved at {self.evaluation_result_file}')
        self.evaluation_visualization()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation')
    # paths
    parser.add_argument(
        '-t',
        '--training_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/training_dataset")
    parser.add_argument(
        '-v',
        '--validation_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/validation_dataset")
    parser.add_argument(
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model_output/20200506-235229")
    parser.add_argument(
        '--eval_result_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/evaluation_results")

    # model parameters
    parser.add_argument('--kernel_dim', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=512)

    args = parser.parse_args()
    validation = ValidationVisualization(args)

    # setup data-loader
    train_dataset = DynamicModelDataset(args.training_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    valid_dataset = DynamicModelDataset(args.validation_data_path)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    for i, (X, y) in enumerate(train_loader):
        logging.info(
            f'training data batch: {i}, input size is {X.size()}, output size is {y.size()}')
        train_labels = y
        break

    for i, (X, y) in enumerate(valid_loader):
        logging.info(
            f'validation data batch: {i}, input size is {X.size()}, output size is {y.size()}')
        validation.validation(X, y, i, train_labels)
        # break for single batch test
        break
