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
from fueling.control.dynamic_model_2_0.conf.model_conf import input_index, output_index, imu_scaling
from fueling.control.dynamic_model_2_0.label_generation.label_generation import generate_mlp_output
from fueling.control.utils.echo_lincoln import echo_lincoln_wrapper
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


class RawDataVisualization():
    """ visualize feature and label """

    def __init__(self, data_file, args):
        self.data_file = data_file
        self.feature = None
        self.data_dim = None
        self.training_data_path = args.training_data_path
        self.plot_path = args.plot_path
        self.model_path = args.dm10_model_path
        # debug
        self.dm_acc = None
        self.dm_w = None
        self.echo_lincoln_acc_w = None
        self.imu_acc = None
        self.imu_w = None
        self.imu_v = []

    def get_data(self):
        """load features"""
        if self.data_file.endswith('.hdf5'):
            feature = read_h5(self.data_file)
        else:
            logging.info('other')
            feature = np.load(self.data_file, allow_pickle=True)
        self.feature = feature[:, :]
        self.data_dim = self.feature.shape[0]
        logging.info(self.data_dim)

    def imu_location(self):
        """ imu location """
        imu_acc = imu_scaling["acc"] * self.acc()
        imu_w = imu_scaling["heading_rate"] * self.feature[:, segment_index["w_z"]]
        self.imu_acc = imu_acc
        self.imu_w = imu_w
        self.imu_v = []
        imu_x, imu_y = self.get_location_from_a_and_w(imu_w, imu_acc, self.imu_v)
        return (imu_x, imu_y)

    def dynamic_model_10_location(self):
        """ get dynamic model 10 predicted location """
        dm_acc, dm_w = self.dynamic_model_10_output()
        self.dm_acc = dm_acc
        self.dm_w = dm_w
        dm_x, dm_y = self.get_location_from_a_and_w(dm_w, dm_acc)
        return (dm_x, dm_y)

    def echo_lincoln_location(self):
        """ get location of rule based echo_lincoln model """
        echo_lincoln_acc_w = echo_lincoln_wrapper(self.data_file)
        self.echo_lincoln_acc_w = echo_lincoln_acc_w
        logging.info(f'echo_lincoln_acc_w size is {echo_lincoln_acc_w.shape}')
        echo_lincoln_x, echo_lincoln_y = self.get_location_from_a_and_w(
            echo_lincoln_acc_w[:, 1], echo_lincoln_acc_w[:, 0])
        return (echo_lincoln_x, echo_lincoln_y)

    def get_location_from_a_and_w(self, ws, accs, v=[]):
        """ integration from acceleration and heading angle change rate to x, y"""
        dt = feature_config["delta_t"]
        theta = self.calc_theta(ws, dt)
        s = self.calc_s(accs, dt, v)
        x = np.zeros((self.data_dim, 1))
        y = np.zeros((self.data_dim, 1))

        # init from GPS location
        x_init = self.feature[0, segment_index['x']]
        y_init = self.feature[0, segment_index['y']]

        for idx in range(0, self.data_dim):
            x[idx] = x_init + s[idx] * np.cos(theta[idx])
            y[idx] = y_init + s[idx] * np.sin(theta[idx])
            # update current location
            x_init = x[idx]
            y_init = y[idx]

        return (x, y)

    def dynamic_model_10_output(self):
        """ dynamic model 1.0 output"""
        # TODO (Shu): merge this with label generatio
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
            mlp_input = np.array([speed, imu_scaling["acc"] * acc,
                                  throttle, brake, steering]).reshape(1, 5)
            predicted_a[k], predicted_w[k] = generate_mlp_output(mlp_input, model, norms)
        logging.debug(f'predicted_a: {predicted_a}')
        logging.debug(f'predicted_w: {predicted_w}')

        return (predicted_a, predicted_w)

    def calc_theta(self, w, dt):
        """ from heading rate to heading"""
        # initial heading
        theta = np.zeros((self.data_dim,))
        init_heading = self._normalize_angle(self.feature[0, segment_index['heading']])
        # init heading from GPS
        theta[0] = init_heading
        logging.info(f'init heading is {theta[0]}')
        for idx in range(1, self.data_dim):
            # theta = theta_0 + omega * dt
            theta[idx] = self._normalize_angle(init_heading + w[idx - 1] * dt)
            # update init heading for next step
            init_heading = theta[idx]
        return theta

    def calc_s(self, acc, dt, v=[]):
        """ s = v0 * dt + 0.5 * a * t * t"""
        # initial velocity
        init_normalized_heading = self._normalize_angle(self.feature[0, segment_index["heading"]])
        v0 = (self.feature[0, segment_index["v_x"]] * np.cos(init_normalized_heading) +
              self.feature[0, segment_index["v_y"]] * np.sin(init_normalized_heading))
        v.append(v0)
        s = np.zeros((self.data_dim, 1))
        for idx in range(1, self.data_dim):
            if v0 + acc[idx - 1] * dt < 0:
                acc[idx - 1] = 0
            s[idx] = self._distance_s(acc[idx - 1], dt, v0)
            v0 = v0 + acc[idx - 1] * dt
            v.append(v0)
        return s

    @staticmethod
    def _distance_s(acc, dt, v0):
        return v0 * dt + 0.5 * acc * dt * dt

    def acc(self):
        """ get ground truth acc """
        acc = np.empty((self.data_dim, 1))
        for idx, heading in enumerate(self.feature[:, segment_index["heading"]]):
            normalized_heading = self._normalize_angle(heading)
            # TODO(Shu): remove this when scaling is added to feature extraction
            acc[idx] = (self.feature[idx, segment_index["a_x"]]
                        * np.cos(normalized_heading) +
                        self.feature[idx, segment_index["a_y"]]
                        * np.sin(normalized_heading))
        return acc

    @staticmethod
    def _normalize_angle(theta):
        # (-pi, pi)
        theta = theta % (2 * math.pi)
        if theta > math.pi:
            theta = theta - 2 * math.pi
        return theta

    def plot(self, imu_only=False):
        """Plot states during the test run"""
        dataset_name = (os.path.basename(self.data_file)).split('.')[0]
        dataset_path = os.path.dirname(self.data_file).replace(
            self.training_data_path, self.plot_path)
        if imu_only:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.set_xlabel('data points', fontdict={'size': 12})
            ax1.set_ylabel('heading angle (rad)', fontdict={'size': 12})
            # ground truth (GPS) heading angle
            ax1.plot(self.feature[:, segment_index['heading']], 'b.', label='GPS')
            # imu heading heading angle
            imu_heading = self.calc_theta(self.imu_w, feature_config["delta_t"])
            ax1.plot(imu_heading, 'r.', label='IMU')
            plt.legend(fontsize=12, numpoints=5, frameon=False)
            plt.title("Heading comparison")
            plt.grid(True)
            # speed comparison
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.set_xlabel('data points', fontdict={'size': 12})
            ax2.set_ylabel('speed (m/s)', fontdict={'size': 12})
            # ground truth (GPS) speed m/s
            ax2.plot(self.feature[:, segment_index['speed']], 'b.', label='GPS')
            # imu speed speed
            ax2.plot(self.imu_v, 'r.', label='IMU')
            plt.legend(fontsize=12, numpoints=5, frameon=False)
            plt.title("GPS and IMU comparison")
            plt.grid(True)
            plt.savefig(self.plot_path + dataset_name + "imu_plot.png")
            plt.show()
        else:
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
            # location for echo lincoln
            echo_lincoln_x, echo_lincoln_y = self.echo_lincoln_location()
            logging.info(imu_x.shape)
            # end pose comparison
            logging.info(
                f'GPS endpose is [{x_position[-1]}, {y_position[-1]}]')
            logging.info(
                f'Dynamic model 1.0 endpose is [{dm_x[-1]}, {dm_y[-1]}]')
            logging.info(
                f'End Pose differences between GPS and dynamic model 1.0 is'
                '[{x_position[-1]-dm_x[-1]}, {y_position[-1]-dm_y[-1]}]')
            plt.plot(x_position - x_position[0], y_position - y_position[0], 'b.', label='GPS')
            plt.plot(imu_x - x_position[0], imu_y - y_position[0], 'r.', label='IMU')
            plt.plot(dm_x - x_position[0], dm_y - y_position[0], 'g.', label="Dynamic model")
            plt.plot(echo_lincoln_x - x_position[0], echo_lincoln_y -
                     y_position[0], 'y.', label='Echo-lincoln')
            plt.plot(0, 0, 'x', markersize=6, color='k')
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
    cur_h5_file = (f'/fuel/fueling/control/dynamic_model_2_0/testdata'
                   '/golden_set/6_2/20190430122402.record.00000.recover_features.npy')
    raw_data_evaluation = RawDataVisualization(cur_h5_file, args)
    raw_data_evaluation.get_data()
    raw_data_evaluation.dynamic_model_10_location()
    raw_data_evaluation.echo_lincoln_location()
    raw_data_evaluation.imu_location()
    plt.plot(raw_data_evaluation.dm_acc, 'b-')
    plt.plot(raw_data_evaluation.echo_lincoln_acc_w[:, 0], 'r-')
    logging.info(raw_data_evaluation.imu_acc.shape)
    plt.plot(raw_data_evaluation.imu_acc, 'y-')
    plt.show()
    plt.plot(raw_data_evaluation.dm_w, 'b-')
    plt.plot(raw_data_evaluation.echo_lincoln_acc_w[:, 1], 'r-')
    plt.plot(raw_data_evaluation.imu_w, 'y-')
    plt.show()
    raw_data_evaluation.plot()
