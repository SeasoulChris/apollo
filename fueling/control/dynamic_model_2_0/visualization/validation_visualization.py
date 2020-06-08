#!/usr/bin/env python
import argparse
import glob
import os
import time


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


from fueling.common import file_utils
from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config
from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset import \
    DynamicModelDataset
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
        self.dst_file_path = args.validation_result_path
        self.validation_result_file = None

        # parameters
        self.kernel_dim = args.kernel_dim
        self.input_dim = feature_config["input_dim"]
        self.output_dim = feature_config["output_dim"]
        self.timestr = time.strftime('%Y%m%d-%H%M%S')

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
        self.model = GPModel(self.inducing_points, encoder_net_model,
                             self.kernel_dim, self.output_dim)
        self.model.load_state_dict(model_state_dict)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.output_dim)
        self.likelihood.load_state_dict(likelihood_state_dict)

    def load_data(self):
        """
        validation data and inducint points
        """
        # features: [num_of_data_points * window_size * input_channel]
        # labels: [same num_of_data_points * output_channel]
        # get inducing points
        self.inducing_points = torch.from_numpy(
            np.load(self.inducing_point_file, allow_pickle=True))
        logging.debug(f'inducing points are {self.inducing_points}')

    def predict(self, input_x):
        """
        make prediction at each point
        """
        logging.debug(f'updated_input is {input_x}')
        self.model.eval()
        self.likelihood.eval()
        # input shape is [100, 1, 6] single data point
        updated_input = torch.transpose(input_x.unsqueeze(0), 0, 1).type(torch.FloatTensor)
        logging.debug(f'updated_input shape is {updated_input.shape}')
        predictions = self.likelihood(self.model(updated_input))
        logging.debug(f'variance is {predictions.variance.detach().numpy()}')
        return predictions.mean.detach().numpy()

    def get_validation_result(self, test_x, test_y, set_id, train_y=None, save_dict=True):
        """
        result for each validation batch
        """
        self.model.eval()
        self.likelihood.eval()
        test_x = torch.transpose(test_x, 0, 1).type(torch.FloatTensor)
        # make prediction with input with uncertainty
        predictions = self.likelihood(self.model(test_x))
        lower, upper = predictions.confidence_region()
        mean = predictions.mean
        variance = predictions.variance
        logging.info(f'mean data shape is {mean.shape}')
        logging.info(f'variance shape is {variance.shape}')
        if save_dict:
            # save result to npy for visualization
            validation_result = dict()
            validation_result['validation_labels'] = test_y.numpy()
            if train_y.numpy().any():
                validation_result['training_labels'] = train_y.numpy()
            validation_result['mean'] = mean.detach().numpy()
            validation_result['variance'] = variance.detach().numpy()
            validation_result['upper'] = upper.detach().numpy()
            validation_result['lower'] = lower.detach().numpy()
            np.save(os.path.join(args.validation_data_path,
                                 f'{set_id}_validation_result.npy'), validation_result)
        return mean.detach().numpy()

    def visualize(self, set_id):
        # get data from .npy file
        logging.info(self.validation_result_file)
        validation_result = np.load(self.validation_result_file, allow_pickle=True).item()

        # ground truth
        validation_labels = validation_result['validation_labels']
        # predicted mean value
        mean = validation_result['mean']
        # confidence region
        upper = validation_result['upper']
        lower = validation_result['lower']
        # training data
        training_labels = validation_result['training_labels']

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
        file_utils.makedirs(os.path.join(self.dst_file_path, self.timestr))
        plt.savefig(os.path.join(self.dst_file_path, self.timestr, f"{set_id}_plot.png"))
        plt.show()

    def validate(self, test_features, test_labels, set_id, train_labels=None, is_plot=True):
        self.load_data()
        logging.info(self.inducing_point_file)
        logging.info(self.inducing_points.shape)
        self.load_model()
        # check if validation result exists
        logging.info(f'Validating set {set_id}')
        self.validation_result_file = os.path.join(args.validation_data_path,
                                                   f'{set_id}_validation_result.npy')
        self.get_validation_result(test_features, test_labels, set_id, train_labels)
        logging.info(f'Validation results are saved at {self.validation_result_file}')
        if is_plot:
            self.visualize(set_id)

    def make_prediction(self, input_x):
        self.load_data()
        logging.info(self.inducing_point_file)
        logging.info(self.inducing_points.shape)
        self.load_model()
        return self.predict(input_x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation')
    # paths
    parser.add_argument(
        '-t',
        '--training_data_path',
        type=str)
    parser.add_argument(
        '-v',
        '--validation_data_path',
        type=str)
    parser.add_argument(
        '--gp_model_path',
        type=str)
    parser.add_argument(
        '--validation_result_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/evaluation_results")

    # model parameters
    parser.add_argument('--kernel_dim', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)

    args = parser.parse_args()
    validator = ValidationVisualization(args)

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

    validator.load_data()
    logging.info(validator.inducing_point_file)
    logging.info(validator.inducing_points.shape)
    validator.load_model()

    # generate batch validation result
    for i, (x_valid_batch, y_valid_batch) in enumerate(valid_loader):
        x_valid_batch = torch.transpose(x_valid_batch, 0, 1)
        pred = validator.likelihood(validator.model(x_valid_batch))
        break

    for i, (X, y) in enumerate(valid_loader):
        logging.debug(
            f'validation data batch: {i}, input size is {X.size()}, output size is {y.size()}')
        for x_data_point, y_data_point in zip(X, y):
            logging.debug(f'Model input is {x_data_point[0,:]}')
            logging.debug(
                f'Model prediction is {validator.predict(x_data_point)} '
                + f'and ground truth is {y_data_point}')
        validator.validate(X, y, i, train_labels)
