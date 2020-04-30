#!/usr/bin/env python
import argparse
import os
import glob


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config
from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
from fueling.control.dynamic_model_2_0.gp_regression.evaluation import evaluation
from fueling.control.dynamic_model_2_0.visualization.raw_data_visualization import RawDataVisualization
import fueling.common.logging as logging


class GoldenSetEvaluation():
    """ golden set evaluation results """

    def __init__(self, golden_set_data, args):
        super().__init__()
        # features numpy file include all paired data points; dimension is 23
        self.features_file_path = golden_set_data  # npy file
        self.features = None
        # TODO(Shu): expand this
        self.args = args
        self.DM10_xy = None
        self.DM20_dx_dy = None
        self.data_frame_length = 100

    def load_data(self):
        logging.info(f'load data from {self.features_file_path}')
        self.features = np.load(self.features_file_path, allow_pickle=True)
        logging.info(f'total data points is {len(self.features)}')

    def get_DM10_result(self, is_save=True):
        raw_data_visualization = RawDataVisualization(self.features_file_path, self.args)
        raw_data_visualization.feature = self.features
        raw_data_visualization.data_dim = self.features.shape[0]
        DM10_x, DM10_y = raw_data_visualization.dynamic_model_10_location()
        self.DM10_xy = np.array(list(zip(DM10_x, DM10_y))).squeeze()
        logging.info(f'Dynamic model 1.0 results\' shape is {self.DM10_xy.shape}')
        logging.info(f'Data points in dynamic model 1.0 looks like {self.DM10_xy[0,:]}')
        if is_save:
            # write to npy
            dst_dir = os.path.dirname(self.features_file_path)
            dst_file = os.path.join(dst_dir, 'DM10_results.npy')
            logging.info(f'Dynamic model 1.0 results are saved to file {dst_file}')
            np.save(dst_file, self.DM10_xy)

    def get_DM20_result(self, is_save=True):
        """ load dynamic model 2.0,
            make prediction,
            and generate pose correction for each point (100:)"""
        dataset = GPDataSet(self.args)
        self.DM20_dx_dy = evaluation(self.args, dataset, GPModel, is_plot=False)
        logging.info(f'Dynamic model 2.0 results\' shape is {self.DM20_dx_dy.shape}')
        logging.info(f'Data points in dynamic model 2.0 looks like {self.DM20_dx_dy[0,:]}')
        if is_save:
            # write to npy
            dst_dir = os.path.dirname(self.features_file_path)
            dst_file = os.path.join(dst_dir, 'DM20_results.npy')
            logging.info(f'Dynamic model 2.0 results are saved to file {dst_file}')
            np.save(dst_file, self.DM20_dx_dy)

    def corrected_DM10_result(self):
        # load dynamic model 2.0 results (dx, dy) for trajectory points from 100 to end
        if self.DM20_dx_dy is None:
            self.DM20_dx_dy = np.load(self.args.dm20_result_path, allow_pickle=True)
        if self.DM10_xy is None:
            self.DM10_xy = np.load(self.args.dm10_result_path, allow_pickle=True)
        # corrected dynamic model 1.0 localization results
        corrected_xy = np.zeros(self.DM10_xy.shape)
        for i in range(0, 2):
            corrected_xy[:, i] = self.DM10_xy[:, i] + \
                self.correction(self.DM20_dx_dy[:, i], self.data_frame_length)
        logging.info(f'Corrected localization matrix has the shape as {corrected_xy.shape}')
        return corrected_xy

    def correction(self, d_value, data_length):
        # pading first (0:99) points with zeros
        padded_d_value = np.pad(d_value, (data_length, 0), 'constant')
        logging.info(
            f'padded array with shape {padded_d_value.shape} and looks like {padded_d_value[0:data_length+2, ]}')
        # composation for dynamic model 1.0 results
        accumulated_value = np.cumsum(padded_d_value, axis=0)
        logging.info(
            f'accumulated array with shape {accumulated_value.shape} and looks like {accumulated_value[0:data_length+2, ]}')
        return accumulated_value

    def plot(self):
        fig, axs = plt.subplots(figsize=[8, 8])
        plt.xlabel('x (m)', fontdict={'size': 12})
        plt.ylabel('y (m)', fontdict={'size': 12})
        plt.axis('equal')
        # location from GPS
        x_position = self.features[:, segment_index['x']]
        y_position = self.features[:, segment_index['y']]
        # location from dynamic model
        if self.DM10_xy is None:
            self.DM10_xy = np.load(self.args.dm10_result_path, allow_pickle=True)
        corrected_xy = self.corrected_DM10_result()
        # shifted position
        plt.plot(x_position - x_position[0], y_position - y_position[0], 'b.', label='GPS')
        plt.plot(self.DM10_xy[:, 0] - self.DM10_xy[0, 0], self.DM10_xy[:, 1] -
                 self.DM10_xy[0, 1], 'g.', label="Dynamic model 1.0")
        plt.plot(corrected_xy[:, 0] - corrected_xy[0, 0], corrected_xy[:, 1] -
                 corrected_xy[0, 1], 'r.', label="Dynamic model 2.0")
        plt.plot(0, 0, 'x', markersize=6, color='k')
        plt.legend(fontsize=12, numpoints=5, frameon=False)
        plt.title("Trajectory Comparison")
        plt.grid(True)
        figure_file = os.path.join(os.path.dirname(self.features_file_path), 'trajectory_plot.png')
        plt.savefig(figure_file)
        logging.info(f'plot is saved at {figure_file}')
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization')
    # paths
    parser.add_argument('-train',
                        '--training_data_path',
                        type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/testdata/0417/train")
    parser.add_argument('-plot',
                        '--plot_path',
                        type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/testdata/plots")
    parser.add_argument('-dm10',
                        '--dm10_model_path', type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/label_generation/mlp_model")

    parser.add_argument('-dm10_result',
                        '--dm10_result_path', type=str)

    parser.add_argument('-dm20_result',
                        '--dm20_result_path', type=str)

    parser.add_argument(
        '-test',
        '--testing_data_path',
        type=str)

    parser.add_argument(
        '-md',
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/testdata/gp_model_output/20200420-214841")

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

    file_path = 'fueling/control/dynamic_model_2_0/testdata/golden_set'
    npy_file_list = glob.glob(os.path.join(file_path, '*/*recover_features.npy'))
    logging.info(f'total {len(npy_file_list )} files: {npy_file_list }')
    args = parser.parse_args()
    for npy_file in npy_file_list:
        logging.info(f'processing npy_file: {npy_file}')
        scenario_id = os.path.dirname(npy_file).split('/')[-1]
        logging.info(scenario_id)
        test_data_folder = os.path.join(
            '/fuel/local_test/labeled_data/2020-04-30-19', scenario_id)
        args.testing_data_path = test_data_folder
        evaluator = GoldenSetEvaluation(npy_file, args)
        evaluator.load_data()
        # if results files (.npy) are provided than skip these two
        evaluator.get_DM10_result()
        evaluator.get_DM20_result()
        evaluator.plot()
        # break
