#!/usr/bin/env python
import argparse
import math
import os

from keras.models import load_model
import h5py
import matplotlib.pyplot as plt
import numpy as np

from fueling.common.h5_utils import read_h5
from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config
from fueling.control.dynamic_model_2_0.conf.model_conf import input_index, output_index
from fueling.control.dynamic_model_2_0.label_generation.label_generation import generate_mlp_output
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


class RawDataVisualization():
    """ visualize feature and label """

    def __init__(self, h5_file, args):
        self.data_file = h5_file
        self.feature = None
        self.data_dim = None
        self.training_data_path = args.training_data_path
        self.plot_path = args.plot_path
        self.model_path = args.dm10_model_path

    def get_data(self):
        self.feature = read_h5(self.data_file)
        self.data_dim = self.feature.shape[0]
        logging.info(self.data_dim)

    def imu_location(self):
        imu_acc = self.acc()
        imu_w = self.feature[:, segment_index["w_z"]]
        imu_x, imu_y = self.get_location_from_a_and_w(imu_w, imu_acc)
        return (imu_x, imu_y)

    def dynamic_model_10_location(self):
        dm_acc, dm_w = self.dynamic_model_10_output()
        dm_x, dm_y = self.get_location_from_a_and_w(dm_w, dm_acc)
        return (dm_x, dm_y)

    def get_location_from_a_and_w(self, ws, accs):
        dt = feature_config["delta_t"]
        theta = self.calc_theta(ws, dt)
        s = self.calc_s(accs, dt)
        x = np.zeros((self.data_dim, 1))
        y = np.zeros((self.data_dim, 1))

        # init from GPS location
        x_init = self.feature[0, segment_index['x']]
        y_init = self.feature[0, segment_index['y']]

        for idx in range(0, self.data_dim):
            # s[0] = 0
            x[idx] = x_init + s[idx] * np.cos(theta[idx])
            y[idx] = y_init + s[idx] * np.sin(theta[idx])
            # update current location
            x_init = x[idx]
            y_init = y[idx]
        return (x, y)

    def dynamic_model_10_output(self):
        # load model parameters
        model_path = self.model_path
        model_norms_path = os.path.join(model_path, 'norms.h5')
        with h5py.File(model_norms_path, 'r') as model_norms_file:
            input_mean = np.array(model_norms_file.get('input_mean'))
            input_std = np.array(model_norms_file.get('input_std'))
            output_mean = np.array(model_norms_file.get('output_mean'))
            output_std = np.array(model_norms_file.get('output_std'))
            norms = (input_mean, input_std, output_mean, output_std)
        model_weights_path = os.path.join(model_path, 'weights.h5')
        model = load_model(model_weights_path)

        # Initialize the first frame's data
        predicted_a = np.zeros((self.data_dim, 1))
        predicted_w = np.zeros((self.data_dim, 1))

        for k in range(0, self.data_dim):
            speed = self.feature[k, segment_index["speed"]]
            acc = self.feature[k, segment_index["a_x"]] * \
                np.cos(self.feature[k, segment_index["heading"]]) + \
                self.feature[k, segment_index["a_y"]] * \
                np.sin(self.feature[k, segment_index["heading"]])
            throttle = self.feature[k, segment_index["throttle"]]
            brake = self.feature[k, segment_index["brake"]]
            steering = self.feature[k, segment_index["steering"]]
            # time delay ?
            mlp_input = np.array([speed, acc, throttle, brake, steering]).reshape(1, 5)
            predicted_a[k], predicted_w[k] = generate_mlp_output(mlp_input, model, norms)
        logging.info(mlp_input.shape)
        return (predicted_a, predicted_w)

    def calc_theta(self, w, dt):
        """ from heading rate to heading"""
        # initial heading
        theta = np.zeros((self.data_dim, 1))
        init_heading = self._normalize_angle(self.feature[0, segment_index['heading']])
        # init heading from GPS
        theta[0] = init_heading
        for idx in range(1, self.data_dim):
            # theta = theta_0 + omega * dt
            theta[idx] = self._normalize_angle(init_heading + w[idx - 1] * dt)
            # update init heading for next step
            init_heading = theta[idx]
        return theta

    def calc_s(self, acc, dt):
        # initial velocity
        init_normalized_heading = self._normalize_angle(self.feature[0, segment_index["heading"]])
        init_v = (self.feature[0, segment_index["v_x"]] * np.cos(init_normalized_heading) +
                  self.feature[0, segment_index["v_y"]] * np.sin(init_normalized_heading))
        v0 = init_v
        s = np.zeros((self.data_dim, 1))
        # logging.info(s.shape)
        for idx in range(1, self.data_dim):
            # logging.info(f'idx: {idx}; acc is {acc[idx]}')
            s[idx] = self._distance_s(acc[idx - 1], dt, v0)
            v0 = v0 + acc[idx - 1] * dt
        return s

    @staticmethod
    def _distance_s(acc, dt, v0):
        return v0 * dt + 1 / 2 * acc * dt * dt

    def acc(self):
        acc = np.empty((self.data_dim, 1))
        for idx, heading in enumerate(self.feature[:, segment_index["heading"]]):
            normalized_heading = self._normalize_angle(heading)
            acc[idx] = (self.feature[idx, segment_index["a_x"]] * np.cos(normalized_heading) +
                        self.feature[idx, segment_index["a_y"]] * np.sin(normalized_heading))
        # logging.info(acc)
        return acc

    @staticmethod
    def _normalize_angle(theta):
        theta = theta % (2 * math.pi)
        if theta > math.pi:
            theta = theta - 2 * math.pi
        return theta

    def plot(self):
        """Plot states during the test run"""
        dataset_name = (os.path.basename(self.data_file)).split('.')[0]
        dataset_path = os.path.dirname(self.data_file).replace(
            self.training_data_path, self.plot_path)
        plt.figure()
        plt.xlabel('x (m)', fontdict={'size': 12})
        plt.ylabel('y (m)', fontdict={'size': 12})
        # location from GPS
        x_position = self.feature[:, segment_index['x']]
        y_position = self.feature[:, segment_index['y']]
        # location from IMU
        imu_x, imu_y = self.imu_location()

        # location from dynamic model
        dm_x, dm_y = self.dynamic_model_10_location()
        plt.plot(x_position, y_position, 'b.', label='GPS')
        plt.plot(imu_x, imu_y, 'r.', label='IMU')
        plt.plot(dm_x, dm_y, 'g.', label="Dynamic model")
        plt.plot(x_position[0], y_position[0], 'x', markersize=6, color='k')
        plt.legend(fontsize=12, numpoints=5, frameon=False)
        plt.title("Trajectory for " + dataset_name)
        plt.grid(True)
        plt.savefig(self.plot_path + dataset_name + "plot.png")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization')
    # paths
    parser.add_argument('-train',
                        '--training_data_path',
                        type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/testdata/train_data")
    parser.add_argument('-plot',
                        '--plot_path',
                        type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/testdata/plots")
    parser.add_argument('-dm10',
                        '--dm10_model_path', type=str,
                        default="/fuel/fueling/control/dynamic_model_2_0/label_generation/mlp_model")
    args = parser.parse_args()
    h5_file_list = []
    for file in file_utils.list_files(args.training_data_path):
        if file.endswith(".hdf5"):
            h5_file_list.append(file)
            logging.info(file)
    cur_h5_file = h5_file_list[0]
    raw_data_evaluation = RawDataVisualization(cur_h5_file, args)
    raw_data_evaluation.get_data()
    raw_data_evaluation.plot()
