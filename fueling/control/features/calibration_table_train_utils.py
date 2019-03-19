#!/usr/bin/env python
import glob
import random
import os

import h5py
import numpy as np

import modules.control.proto.calibration_table_pb2 as calibration_table_pb2

from fueling.control.features.filters import Filters
from fueling.control.features.neural_network_tf import NeuralNetworkTF
import fueling.common.colored_glog as glog
import fueling.common.file_utils as file_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils

def choose_data_file(elem, vehicle_type, brake_or_throttle, train_or_test):
    # TODO: Not record_dir.
    record_dir = elem[0]
    hdf5_file = glob.glob(
        # TODO: Please write detailed document under control/calibration_table, about the file tree
        # structure. As the logic has really strict requirement on how the data is organized.
        '{}/{}/{}/{}/*.hdf5'.format(record_dir, vehicle_type, brake_or_throttle, train_or_test))
    return (elem[0], hdf5_file)


def generate_segments(h5s):
    segments = []
    for h5 in h5s:
        # TODO: Use glog.
        print('Loading {}'.format(h5))
        # TODO: Avoid single-char variable name.
        with h5py.File(h5, 'r+') as f:
            # TODO: Simplify the logic: segments = [np.array(segment) for segment in f.itervalues()]
            names = [n for n in f.keys()]
            print('f.keys', f.keys())
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
    # TODO: Looping a "range(len(segments))" equals looping segments directly.
    for i in range(len(segments)):
        total_len += segments[i].shape[0]
    print("total_len = ", total_len)
    dim_input = 2
    dim_output = 1
    X = np.zeros([total_len, dim_input])
    Y = np.zeros([total_len, dim_output])
    i = 0
    for j in range(len(segments)):
        segment = segments[j]
        for k in range(1, segment.shape[0]):
            X[i, 0:2] = segment[k, [0, 2]]  # speed & cmd
            Y[i, 0] = segment[k, 1]  # acc
            i += 1
    return X, Y

# TODO: Avoid naming like "elem" which has no information. Describe what it is.
def train_model(elem, layer, train_alpha):
    """
    train model
    """
    # TODO: Extract tuples in one go: (X_train, Y_train), (X_test, Y_test) = elem
    X_train = elem[0][0]
    Y_train = elem[0][1]
    X_test = elem[1][0]
    Y_test = elem[1][1]

    model = NeuralNetworkTF(layer)
    params, train_cost, test_cost = model.train(X_train, Y_train, X_test, Y_test,
                                                alpha=train_alpha, print_loss=True)
    glog.info(" model train cost: %f" % train_cost)
    glog.info(" model test cost: %f " % test_cost)
    return model


def write_table(elem,
                speed_min, speed_max, speed_segment_num,
                axis_cmd_min, axis_cmd_max, cmd_segment_num,
                table_filename):
    """
    write calibration table
    """
    model = elem[1]
    calibration_table_pb = calibration_table_pb2.ControlCalibrationTable()

    speed_array = np.linspace(speed_min, speed_max, num=speed_segment_num)
    cmd_array = np.linspace(axis_cmd_min, axis_cmd_max, num=cmd_segment_num)

    speed_array, cmd_array = np.meshgrid(speed_array, cmd_array)
    grid_array = np.array([[s, c] for s, c in zip(np.ravel(speed_array), np.ravel(cmd_array))])
    acc_array = model.predict(grid_array).reshape(speed_array.shape)

    for cmd_index in range(cmd_segment_num):
        for speed_index in range(speed_segment_num):
            item = calibration_table_pb.calibration.add()
            item.speed = speed_array[cmd_index][speed_index]
            item.command = cmd_array[cmd_index][speed_index]
            item.acceleration = acc_array[cmd_index][speed_index]

    path = elem[0]

    calibration_table_utils.write_h5_cal_tab(acc_array, path, table_filename)

    with open(os.path.join(path, table_filename), 'w') as wf:
        wf.write(str(calibration_table_pb))
    return table_filename
