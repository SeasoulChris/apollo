
import glob
import h5py

import numpy as np

from neural_network_tf import NeuralNetworkTF
import modules.control.proto.calibration_table_pb2 as calibration_table_pb2


class LonAutoCalibrationTF(object):
    """
    Longitudinal Auto-Calibration with tensorflow
    """

    def __init__(self, mode):
        """
        init function
        """
        # select mode
        self.mode = mode
        if mode == 'throttle':
            self.traindata_filename = "throttle_train"
            self.testdata_filename = "throttle_test"
            self.table_filename = "mkz_throttle_calibration_table.pb.txt"
            # 5.0~30.0
            self.axis_cmd_min = 5.0  # Transit # 18.0 Mkz7
            self.axis_cmd_max = 30.0  # Transit # 60.0 Mkz7
            self.layer = [2, 15, 1]
            self.alpha = 0.05
        elif mode == 'brake':
            self.traindata_filename = "brake_train"
            self.testdata_filename = "brake_test"
            self.table_filename = "mkz_brake_calibration_table.pb.txt"
            # Transit-30.0 ~ -7.0
            self.axis_cmd_min = -30.0  # -35.0 Mkz7
            self.axis_cmd_max = -7.0  # -21.0 Mkz7
            self.layer = [2, 10, 1]
            self.alpha = 0.05
        else:
            print("please use correct mode")
            sys.exit(-1)

        # # load data
        # try:
        #     data_train = np.loadtxt(self.traindata_filename)
        #     self.X_train = np.array(data_train[..., 0: 2])
        #     self.Y_train = np.array(data_train[..., 2]).reshape(-1, 1)
        #     data_test = np.loadtxt(self.testdata_filename)
        #     self.X_test = np.array(data_test[..., 0: 2])
        #     self.Y_test = np.array(data_test[..., 2]).reshape(-1, 1)
        # except Exception:
        #     print("please generate feature first")
        #     sys.exit(-1)

        self.speed_min = 0.0
        self.speed_max = 20.0
        self.cmd_segment_num = 10
        self.speed_segment_num = 50

        # self.fig = plt.figure()
        # self.ax = self.fig.gca(projection='3d')
        self.model = NeuralNetworkTF(self.layer)


def generate_segments(h5s):
    segments = []
    for h5 in h5s:
        print('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as f:
            names = [n for n in f.keys()]
            if len(names) < 1:
                continue
            for i in range(len(names)):
                ds = np.array(f[names[i]])
                segments.append(ds)
    # shuffle(segments)
    print('Segments count: ', len(segments))
    return segments


def generate_data(segments):
    """ combine data from each segments """
    total_len = 0
    for i in range(len(segments)):
        total_len += segments[i].shape[0]
    print "total_len = ", total_len
    dim_input = 2
    dim_output = 1
    X = np.zeros([total_len, dim_input])
    Y = np.zeros([total_len, dim_output])
    i = 0
    for j in range(len(segments)):
        segment = segments[j]
        for k in range(segment.shape[0]):
            if k > 0:
                X[i, 0:2] = segment[k, 0:2]
                Y[i, 0] = segment[k, 2]
                i += 1
    return X, Y


def train_model(obj):
    """
    train model
    """
    params, train_cost, test_cost = obj.model.train(X_train, Y_train,
                                                    X_test, Y_test,
                                                    alpha=obj.alpha, print_loss=True)
    print(mode + " model train cost: " + str(train_cost))
    print(mode + " model test cost: " + str(test_cost))


def find_best_model(obj):
    """
    find best model
    """
    min_cost = 9999
    alphas = [0.001, 0.01, 0.05, 0.1]
    w_lambdas = [0.001, 0.01, 0.1]
    layers = [[2, 10, 1], [2, 15, 1], [2, 20, 1], [2, 25, 1]]
    for layer in layers:
        for alpha in alphas:
            for w_lambda in w_lambdas:
                obj.model = NeuralNetworkTF(layer)
                params, train_cost, test_cost = obj.model.train(X_train, Y_train,
                                                                X_test, Y_test,
                                                                alpha=alpha, w_lambda=w_lambda)
                if test_cost < min_cost:
                    min_cost = test_cost
                    best_alpha = alpha
                    best_w_lambda = w_lambda
                    best_layer = layer
                print("test cost: " + str(test_cost) + ", layer: " + str(layer) +
                      ", learning rate: " + str(alpha) + ", w_lambda: " + str(w_lambda))
    print("min cost: " + str(min_cost))
    print("best layer: " + str(best_layer))
    print("best learning rate: " + str(best_alpha))
    print("best w_lambda: " + str(best_w_lambda))


def write_table(self):
    """
    write calibration table
    """
    calibration_table_pb = calibration_table_pb2.ControlCalibrationTable()
    speed_array = np.linspace(
        self.speed_min, self.speed_max, num=self.speed_segment_num)
    cmd_array = np.linspace(
        self.axis_cmd_min, self.axis_cmd_max, num=self.cmd_segment_num)
    speed_array, cmd_array = np.meshgrid(speed_array, cmd_array)
    grid_array = np.array([[s, c] for s, c in zip(
        np.ravel(speed_array), np.ravel(cmd_array))])
    acc_array = self.model.predict(grid_array).reshape(speed_array.shape)

    for cmd_index in range(self.cmd_segment_num):
        for speed_index in range(self.speed_segment_num):
            item = calibration_table_pb.calibration.add()
            item.speed = speed_array[cmd_index][speed_index]
            item.command = cmd_array[cmd_index][speed_index]
            item.acceleration = acc_array[cmd_index][speed_index]

    with open(self.table_filename, 'w') as wf:
        wf.write(str(calibration_table_pb))


if __name__ == '__main__':

    FLAGS_find_best_model = False
    mode = "throttle"
    obj = LonAutoCalibrationTF(mode)

    hdf5_train = glob.glob(
        '/apollo/modules/data/fuel/testdata/control/{}/*.hdf5'.format(obj.traindata_folder))
    segments_train = generate_segments(hdf5_train)
    X_train, Y_train = generate_data(segments_train)

    hdf5_test = glob.glob(
        '/apollo/modules/data/fuel/testdata/control/{}/*.hdf5'.format(obj.testdata_folder))
    segments_test = generate_segments(hdf5_test)
    X_test, Y_test = generate_data(segments_test)

    if FLAGS_find_best_model:
        find_best_model(obj)
    else:
        train_model(obj)
        write_table(obj)
