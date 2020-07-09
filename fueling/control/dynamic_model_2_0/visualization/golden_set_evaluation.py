#!/usr/bin/env python
import argparse
import glob
import math
import os


from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.spatial import distance
import gpytorch
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, imu_scaling
from fueling.control.dynamic_model_2_0.conf.model_conf import \
    feature_config, input_index, output_index
from fueling.control.dynamic_model_2_0.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
from fueling.control.dynamic_model_2_0.visualization.raw_data_visualization import \
    RawDataVisualization
from fueling.control.dynamic_model_2_0.visualization.validation_visualization import \
    ValidationVisualization
from fueling.control.dynamic_model_2_0.label_generation.label_generation import generate_gp_data
import fueling.common.h5_utils as h5_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


class GoldenSetEvaluation():
    """ golden set evaluation results """

    def __init__(self, feature_dir, standardization_factors_file, args, model_id):
        super().__init__()
        # features numpy file include all paired data points; dimension is 23
        self.feature_dir = feature_dir
        # make dir
        self.result_folder = os.path.join(feature_dir, model_id)
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.features_file_path = os.path.join(self.feature_dir, 'features.npy')
        self.DM10_result_file = os.path.join(feature_dir, 'DM10_results.npy')
        self.echo_lincoln_result_file = os.path.join(feature_dir, 'Echo_Lincoln_results.npy')
        self.DM20_result_file = os.path.join(self.result_folder, 'DM20_results.py')
        self.analyses_result_file = os.path.join(self.result_folder, 'analyses_result.py')

        self.features = None
        self.get_non_overlapping_features()
        # TODO(Shu): expand this
        self.args = args
        self.DM10_xy = None
        self.DM20_dx_dy = None
        self.echo_lincoln_xy = None
        self.data_frame_length = 100
        self.raw_data_visualization = None
        self.imu_xy = None
        self.DM20 = ValidationVisualization(self.args)
        # get inducing points for model initialization
        self.DM20.load_data()  # inducing points
        self.DM20.load_model()
        self.get_DM20_normalization_factors(standardization_factors_file)
        self.label = None
        self.lower = None
        self.upper = None

    def get_DM20_normalization_factors(self, standardization_factors_file):
        self.standardization_factors = np.load(
            standardization_factors_file, allow_pickle=True).item()

    def standardize(self, inputs):
        """Standardize given data"""
        input_mean = self.standardization_factors['mean']
        input_std = self.standardization_factors['std']
        inputs_standardized = (inputs - input_mean) / input_std
        return inputs_standardized

    def get_non_overlapping_features(self, non_overlapping_features=False):
        # features file are 100 frame with 99 frame overlapping
        hdf5_files = file_utils.list_files_with_suffix(self.feature_dir, '.hdf5')

        def get_file_number(hdf5_files):
            """ files are named in time sequence """
            file_name = os.path.basename(hdf5_files).split('.')[0]
            return int(file_name.split('-')[0])

        self.hdf5_files = sorted(hdf5_files, key=get_file_number)
        for idx, hdf5_file in enumerate(self.hdf5_files):
            if self.features is None:
                # 100 * 23
                self.features = h5_utils.read_h5(hdf5_file)
            else:
                # [feature, 1*23]
                updated_features = h5_utils.read_h5(hdf5_file)
                if non_overlapping_features:
                    self.features = np.concatenate(
                        (self.features, updated_features), axis=0)
                else:
                    self.features = np.concatenate(
                        (self.features, np.expand_dims(updated_features[-1, :], axis=0)), axis=0)
        logging.info(self.features.shape)
        np.save(self.features_file_path, self.features)

    def get_dm20_input(self, hdf5_file):
        features = h5_utils.read_h5(hdf5_file)
        input_segment = np.zeros((feature_config["input_window_size"], feature_config["input_dim"]))
        input_segment[:, input_index["v"]] = features[:, segment_index["speed"]]
        # add scaling to acceleration
        input_segment[:, input_index["a"]] = \
            imu_scaling["acc"] * features[:, segment_index["a_x"]] \
            * np.cos(features[:, segment_index["heading"]]) \
            + features[:, segment_index["a_y"]] \
            * np.sin(features[:, segment_index["heading"]])
        input_segment[:, input_index["u_1"]] = features[:, segment_index["throttle"]]
        input_segment[:, input_index["u_2"]] = features[:, segment_index["brake"]]
        input_segment[:, input_index["u_3"]] = features[:, segment_index["steering"]]
        input_segment[:, input_index["phi"]] = features[:, segment_index["heading"]]
        _, dm01_output_segment = generate_gp_data(
            self.args.dm10_model_path, features.copy())
        return input_segment, np.expand_dims(dm01_output_segment, axis=0)

    def load_data(self):
        logging.info(f'total data points is {len(self.features)}')
        self.raw_data_visualization = RawDataVisualization(self.features_file_path, self.args)
        self.raw_data_visualization.feature = self.features
        self.raw_data_visualization.data_dim = self.features.shape[0]

    def get_imu_result(self):
        imu_data = RawDataVisualization(self.features_file_path, self.args)
        imu_data.feature = self.features
        imu_data.data_dim = self.features.shape[0]
        self.imu_xy = imu_data.imu_location()
        self.imu_acc = imu_data.acc()

    def get_echo_lincoln_result(self, is_save=True):
        echo_lincoln_x, echo_lincoln_y = self.raw_data_visualization.echo_lincoln_location()
        self.echo_lincoln_xy = np.array(list(zip(echo_lincoln_x, echo_lincoln_y))).squeeze()
        logging.debug(f'Echo Lincoln results\' shape is {self.echo_lincoln_xy}')
        logging.debug(f'Data points in Echo Lincoln looks like {self.echo_lincoln_xy[0,:]}')
        if is_save:
            # write to npy
            logging.info(f'Echo Lincoln results are saved to file {self.echo_lincoln_result_file}')
            np.save(self.echo_lincoln_result_file, self.echo_lincoln_xy)

    def get_DM10_result(self, is_save=True):
        DM10_x, DM10_y = self.raw_data_visualization.dynamic_model_10_location()
        self.DM10_xy = np.array(list(zip(DM10_x, DM10_y))).squeeze()
        logging.debug(f'Dynamic model 1.0 results\' shape is {self.DM10_xy.shape}')
        logging.debug(f'Data points in dynamic model 1.0 looks like {self.DM10_xy[0,:]}')
        if is_save:
            # write to npy
            logging.info(f'Dynamic model 1.0 results are saved to file {self.DM10_result_file}')
            np.save(self.DM10_result_file, self.DM10_xy)

    def get_DM10_result_in_DM20(self, DM10_in_DM20, features, updated_loc):
        DM10_in_DM20.feature = features
        DM10_in_DM20.data_dim = self.data_frame_length
        # get self.data_frame_length input and starting point is correct location
        # (dm10_x, dm10_y)
        return DM10_in_DM20.dynamic_model_10_location(updated_loc)

    def get_DM20_result_from_features(self):
        # read features
        dm20_hdf5_file = self.hdf5_files
        for idx, hdf5_file in enumerate(dm20_hdf5_file):
            # generate dm2.0 input
            input_segment, dm01_output_segment = self.get_dm20_input(hdf5_file)
            # normalized input_segment
            input_segment = torch.from_numpy(self.standardize(input_segment))
            # generate predicted results
            predict_result, lower, upper = self.DM20.predict(input_segment)

            if idx == 0:
                self.DM20_dx_dy = predict_result
                self.label = dm01_output_segment
                self.lower = lower
                self.upper = upper
            else:
                self.DM20_dx_dy = np.concatenate((self.DM20_dx_dy, predict_result), axis=0)
                self.label = np.concatenate((self.label, dm01_output_segment), axis=0)
                self.lower = np.concatenate((self.lower, lower), axis=0)
                self.upper = np.concatenate((self.upper, upper), axis=0)
        dst_label_file = os.path.join(self.result_folder, 'label.npy')
        np.save(dst_label_file, self.label)
        dst_DM20_dxdy_file = os.path.join(self.result_folder, 'DM20_dxdy.npy')
        np.save(dst_DM20_dxdy_file, self.DM20_dx_dy)
        np.save(os.path.join(self.result_folder, 'lower.npy'), self.lower)
        np.save(os.path.join(self.result_folder, 'upper.npy'), self.upper)

    def correct_non_overlap_data(self):
        DM10_in_DM20 = RawDataVisualization(self.features_file_path, self.args)
        gp_DM10_in_DM20 = RawDataVisualization(self.features_file_path, self.args)
        # compasant every 100 frames
        if self.label is None:
            self.label = np.load(os.path.join(self.result_folder, 'label.npy'), allow_pickle=True)
        # correction value for every 100 frames
        d_correction = self.label[::self.data_frame_length, :]
        logging.debug(f'model label is {d_correction}')
        # gp result
        if self.DM20_dx_dy is None:
            self.DM20_dx_dy = np.load(os.path.join(
                self.result_folder, 'DM20_dxdy.npy'), allow_pickle=True)
        # correction value for every 100 frames
        gp_d_correction = self.DM20_dx_dy[::self.data_frame_length, :]
        logging.debug(f'model output is {gp_d_correction}')
        # loop over feature files
        # get DM10 output with updated pos
        dm20_hdf5_file = self.hdf5_files[::100]
        updated_loc = None
        gp_updated_loc = None
        if self.DM10_xy is None:
            self.DM10_xy = np.load(self.DM10_result_file, allow_pickle=True)
        self.label_corrected_xy = np.expand_dims(
            self.DM10_xy[0, :], axis=0)  # first point is not corrected
        self.gp_corrected_xy = np.expand_dims(self.DM10_xy[0, :], axis=0)

        for idx, hdf5_file in enumerate(dm20_hdf5_file):
            features = h5_utils.read_h5(hdf5_file)
            # label compansation
            dm_x, dm_y = self.get_DM10_result_in_DM20(DM10_in_DM20, features, updated_loc)
            gp_dm_x, gp_dm_y = self.get_DM10_result_in_DM20(
                gp_DM10_in_DM20, features, gp_updated_loc)
            updated_loc = np.transpose(np.array([dm_x[-1] + d_correction[idx, 0],
                                                 dm_y[-1] + d_correction[idx, 1]]))
            # gp correction
            gp_updated_loc = np.transpose(np.array([gp_dm_x[-1] + gp_d_correction[idx, 0],
                                                    gp_dm_y[-1] + gp_d_correction[idx, 1]]))
            self.label_corrected_xy = np.concatenate(
                (self.label_corrected_xy, updated_loc), axis=0)
            self.gp_corrected_xy = np.concatenate(
                (self.gp_corrected_xy, gp_updated_loc), axis=0)
        logging.info(self.DM20_result_file)
        np.save(self.DM20_result_file, self.gp_corrected_xy)

    def get_error_analyses(self):
        # load data
        # ground truth
        self.features = np.load(self.features_file_path, allow_pickle=True)
        xy_position = self.features[:, segment_index['x']:segment_index['y'] + 1]
        logging.info(xy_position.shape)
        # location from dynamic model 1.0
        if self.DM10_xy is None:
            self.DM10_xy = np.load(self.DM10_result_file, allow_pickle=True)
        # location from echo Lincoln model
        if self.echo_lincoln_xy is None:
            self.echo_lincoln_xy = np.load(self.echo_lincoln_result_file, allow_pickle=True)
        logging.info(self.echo_lincoln_xy.shape)
        # get location of DM20 model
        if self.gp_corrected_xy is None:
            self.gp_corrected_xy = np.load(self.analyses_result_file, allow_pickle=True)

        # 1s, 10s, 30s and end of trajectory error
        time_frame_ranges = [99, 999, min(
            2999, self.features.shape[0] - 1), self.features.shape[0] - 1]
        analyses_result = dict()
        analyses_result["echo_lincoln_cates"] = []
        analyses_result["echo_lincoln_mates"] = []
        analyses_result["DM10_cates"] = []
        analyses_result["DM10_mates"] = []
        analyses_result["DM20_cates"] = []
        analyses_result["DM20_mates"] = []
        for time_frame in time_frame_ranges:
            echo_lincoln_mate = self.calc_mate(self.echo_lincoln_xy, xy_position, time_frame)
            analyses_result["echo_lincoln_mates"].append(echo_lincoln_mate)
            logging.info(f'Echo_lincoln: {time_frame}th time frame Mate is {echo_lincoln_mate:.4f}')
            echo_lincoln_cate = self.calc_cate(self.echo_lincoln_xy, xy_position, time_frame)
            analyses_result["echo_lincoln_cates"].append(echo_lincoln_cate)
            logging.info(f'Echo_lincoln: {time_frame}th time frame Cate is {echo_lincoln_cate:.4f}')

            DM10_mate = self.calc_mate(self.DM10_xy, xy_position, time_frame)
            analyses_result["DM10_mates"].append(DM10_mate)
            logging.info(f'DM10: {time_frame}th time frame Mate is {DM10_mate:.4f}')
            DM10_cate = self.calc_cate(self.DM10_xy, xy_position, time_frame)
            analyses_result["DM10_cates"].append(DM10_cate)
            logging.info(f'DM10: {time_frame}th time frame Cate is {DM10_cate:.4f}')

            DM20_mate = (self.calc_mate(
                self.gp_corrected_xy, xy_position[::self.data_frame_length, :],
                min(int(time_frame / self.data_frame_length + 1),
                    self.gp_corrected_xy.shape[0] - 1)) / self.data_frame_length)
            analyses_result["DM20_mates"].append(DM20_mate)
            logging.info(f'DM20: {time_frame}th time frame Mate is {DM20_mate:.4f}')
            DM20_cate = self.calc_cate(
                self.gp_corrected_xy, xy_position[::self.data_frame_length, :],
                min(int(time_frame / self.data_frame_length + 1),
                    self.gp_corrected_xy.shape[0] - 1))
            analyses_result["DM20_cates"].append(DM20_cate)
            logging.info(f'DM20: {time_frame}th time frame Cate is {DM20_cate:.4f}')

        np.save(self.analyses_result_file, analyses_result)

    def calc_mate(self, predicted_result, ground_truth, nth_frame):
        """  Mean Absolute Trajectory Error (m-ATE):
        The mean absolute trajectory error averages the magnitude of
        translational error of estimated poses with respect to a ground truth trajectory defined
        within the same navigation frame."""
        # N-th frame (N = 1 for Cate; N = N-th frame for Mate)
        # predicted_result np.array with shape 1 * M_dim
        # ground_truth with same shape
        return (math.sqrt(np.sum((predicted_result[nth_frame, :] - ground_truth[nth_frame, :])**2))
                / nth_frame)

    def calc_cate(self, predicted_result, ground_truth, nth_frame):
        """  Cumulative Absolute Trajectory Error (c-ATE):
        Cumulative absolute trajectory error sums translational
        em-ATE up to a given point in a trajectory."""
        # N-th frame (N = 1 for Cate; N = N-th frame for Mate)
        # predicted_result np.array with shape 1 * M_dim
        # ground_truth with same shape
        return math.sqrt(np.sum((predicted_result[nth_frame, :] - ground_truth[nth_frame, :])**2))

    def plot_IMU(self):
        logging.info(self.features.shape)
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        plt.xlabel('x (m)', fontdict={'size': 12})
        plt.ylabel('y (m)', fontdict={'size': 12})
        plt.axis('equal')
        # location from GPS
        x_position = self.features[:, segment_index['x']]
        y_position = self.features[:, segment_index['y']]
        # log info of accumulated error
        logging.info(
            f'GPS data shape is {len(x_position)} and first data point is'
            + '{x_position[0]}, {y_position[0]}')
        logging.info(
            f'IMU data shape is {self.imu_xy[0].shape[0]} and first data point is'
            + f'{self.imu_xy[0][0]}, {self.imu_xy[1][0]}')
        plt.plot(x_position[::100] - x_position[0],
                 y_position[::100] - y_position[0], 'b.', label='GPS')
        plt.plot(self.imu_xy[0][::100] - self.imu_xy[0][0], self.imu_xy[1][::100]
                 - self.imu_xy[1][0], 'm.', label='IMU')

        plt.plot(0, 0, 'x', markersize=6, color='k')
        plt.legend(fontsize=12, numpoints=5, frameon=False)
        plt.title("Trajectory Comparison")
        plt.grid(True)
        ax2 = fig.add_subplot(2, 1, 2)
        normalized_heading_angle = []
        for heading_angle in self.features[:200, 0]:
            normalized_heading_angle.append(self._normalize_angle(heading_angle))
        logging.info(self.imu_acc.shape)
        logging.info(self.imu_acc[1, :])
        plt.plot(self.imu_acc[:200, 0], 'g.', label='heading angle')
        figure_file = os.path.join(os.path.dirname(
            self.features_file_path), 'cmp_GPS_imu_multi_plot.png')
        plt.savefig(figure_file)
        logging.info(f'plot is saved at {figure_file}')
        plt.show()

    def _normalize_angle(self, theta):
        theta = theta % (2 * math.pi)
        if theta > math.pi:
            theta = theta - 2 * math.pi
        return theta

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
        # location from echo Lincoln model
        if self.echo_lincoln_xy is None:
            logging.info(self.echo_lincoln_result_file)
            self.echo_lincoln_xy = np.load(self.echo_lincoln_result_file, allow_pickle=True)
        if self.lower is None:
            self.lower = np.load(os.path.join(self.result_folder, 'lower.npy'), allow_pickle=True)
            self.upper = np.load(os.path.join(self.result_folder, 'upper.npy'), allow_pickle=True)
        # log info of accumulated error
        xy_position = self.features[:, segment_index['x']:segment_index['y'] + 1]
        logging.info(f'Ground truth trajectory shape is : {xy_position.shape}')
        logging.info(f'Dynamic model 1.0 trajectory shape is :{self.DM10_xy.shape}')
        logging.info(f'Echo Lincoln trajectory shape is :{self.echo_lincoln_xy.shape}')
        DM10_error = self.calc_accumulated_error(xy_position, self.DM10_xy)
        echo_lincoln_error = self.calc_accumulated_error(xy_position, self.echo_lincoln_xy)
        DM20_error = self.calc_accumulated_error(self.gp_corrected_xy, self.label_corrected_xy)
        logging.info(f'Dynamic model 1.0 accumulated error is :{DM10_error} m')
        logging.info(f'Echo_lincoln accumulated error is :{echo_lincoln_error} m')
        # shifted position
        plt.plot(x_position - x_position[0], y_position - y_position[0], 'b.', label='GPS')
        plt.plot(self.imu_xy[0] - self.imu_xy[0][0], self.imu_xy[1]
                 - self.imu_xy[1][0], 'm.', label='IMU')
        plt.plot(self.DM10_xy[::100, 0] - self.DM10_xy[0, 0], self.DM10_xy[::100, 1]
                 - self.DM10_xy[0, 1], 'g.',
                 label=f"Dynamic model 1.0, error is {DM10_error:.3f} m")
        plt.plot(self.echo_lincoln_xy[:, 0] - self.echo_lincoln_xy[0, 0],
                 self.echo_lincoln_xy[:, 1] - self.echo_lincoln_xy[0, 1],
                 'y.', label=f"Echo Lincoln, error is {echo_lincoln_error:.3f} m")
        # correct DM1.0 result (using label)
        plt.plot(self.label_corrected_xy[:, 0] - self.DM10_xy[0, 0],
                 self.label_corrected_xy[:, 1] - self.DM10_xy[0, 1], 'k.',
                 label='Corrected result with label')
        # correct DM1.0 result (using gp model)
        axs.errorbar(self.gp_corrected_xy[1:, 0] - self.DM10_xy[0, 0],
                     self.gp_corrected_xy[1:, 1] - self.DM10_xy[0, 1],
                     yerr=[abs(self.lower[::100, 1]), self.upper[::100, 1]],
                     xerr=[abs(self.lower[::100, 0]), self.upper[::100, 0]], fmt='rx', capsize=5,
                     label=f'Corrected result with GP model, error is {DM20_error:.3f} m')
        plt.plot(0, 0, 'x', markersize=6, color='k')
        plt.legend(fontsize=12, numpoints=5, frameon=False)
        plt.title("Trajectory Comparison")
        plt.grid(True)
        figure_file = os.path.join(self.result_folder, 'trajectory_plot_scaled_imu.png')
        plt.savefig(figure_file)
        logging.info(f'plot is saved at {figure_file}')
        plt.show()

    def calc_accumulated_error(self, ref_trajectory, actual_trajectory):
        """ ref_trajectory[num_of_data_points, dim_of point]"""
        num_points = min(ref_trajectory.shape[0], actual_trajectory.shape[0])
        # distance (error)
        ref_coords = ref_trajectory[: num_points, :]
        actual_coords = actual_trajectory[: num_points, :]
        errors = distance.cdist(ref_coords, actual_coords, 'euclidean').diagonal()
        # RMS of distance
        return math.sqrt(np.mean(np.square(errors)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization')
    # paths
    parser.add_argument('-train',
                        '--training_data_path',
                        type=str)
    parser.add_argument('-v',
                        '--validation_data_path',
                        type=str)
    parser.add_argument('--validation_result_path',
                        type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/"
                                + "testdata/evaluation_results")
    parser.add_argument('-plot',
                        '--plot_path',
                        type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/testdata/plots")
    parser.add_argument('-dm10',
                        '--dm10_model_path', type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/"
                                + "label_generation/mlp_model")

    parser.add_argument('-dm10_result',
                        '--dm10_result_path', type=str)

    parser.add_argument('-dm20_result',
                        '--dm20_result_path', type=str)

    parser.add_argument('--echo_lincoln_result_path', type=str)

    parser.add_argument(
        '-test',
        '--testing_data_path',
        type=str)

    parser.add_argument(
        '-md',
        '--gp_model_path',
        type=str)

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=128)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    # argument to train or test gp
    parser.add_argument('--train_gp', type=bool, default=True)
    parser.add_argument('--test_gp', type=bool, default=True)

    # argument to use cuda or not
    parser.add_argument('--use_cuda', type=bool, default=False)

    # golden set file path
    parser.add_argument('--golden_set_data_dir', type=str)
    # training data standardization_factors file path
    parser.add_argument('--normalization_factor_file_path', type=str)

    args = parser.parse_args()
    model_id = '20200706-1722'
    # loop over each golden set scenarios
    evaluator = GoldenSetEvaluation(args.golden_set_data_dir,
                                    args.normalization_factor_file_path, args, model_id)
    evaluator.load_data()
    evaluator.get_imu_result()
    evaluator.plot_IMU()
    evaluator.get_DM10_result()
    evaluator.get_echo_lincoln_result()
    evaluator.get_DM20_result_from_features()
    evaluator.correct_non_overlap_data()
    evaluator.get_error_analyses()
    evaluator.plot()
