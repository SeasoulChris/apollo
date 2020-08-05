#!/usr/bin/env python
import os

import h5py
import numpy as np

import modules.control.proto.calibration_table_pb2 as calibration_table_pb2

from fueling.control.features.neural_network_tf import NeuralNetworkTF
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils


def generate_segments(h5s):
    segments = []
    for h5 in h5s:
        logging.info('Loading %s' % str(h5))
        with h5py.File(h5, 'r+') as fin:
            segments.extend(map(np.array, fin.values()))
    # shuffle(segments)
    print('Segments count: ', len(segments))
    return segments


def generate_data(segments):
    """ combine data from each segments """
    total_len = 0
    for segment in segments:
        total_len += segment.shape[0]
    print("total_len = ", total_len)
    dim_input = 2
    dim_output = 1
    X = np.zeros([total_len, dim_input])
    Y = np.zeros([total_len, dim_output])
    i = 0
    for segment in segments:
        for k in range(1, segment.shape[0]):
            X[i, 0:2] = segment[k, [0, 2]]  # speed & cmd
            Y[i, 0] = segment[k, 1]  # acc
            i += 1
    return X, Y


def train_model(data_sets, layer, train_alpha):
    """
    train model
    """
    (X_train, Y_train), (X_test, Y_test) = data_sets

    model = NeuralNetworkTF(layer)
    _, train_cost, test_cost = model.train(X_train, Y_train, X_test, Y_test,
                                           alpha=train_alpha, print_loss=True)
    logging.info(" model train cost: %f" % train_cost)
    logging.info(" model test cost: %f " % test_cost)
    return model


def write_table(model, target_dir,
                speed_min, speed_max, speed_segment_num,
                axis_cmd_min, axis_cmd_max, cmd_segment_num,
                table_filename):
    """
    write calibration table
    """
    calibration_table_pb = calibration_table_pb2.ControlCalibrationTable()

    speed_array = np.linspace(speed_min, speed_max, num=speed_segment_num)
    cmd_array = np.linspace(axis_cmd_min, axis_cmd_max, num=cmd_segment_num)

    speed_array, cmd_array = np.meshgrid(speed_array, cmd_array)  # col, row
    grid_array = np.array([[s, c] for s, c in zip(np.ravel(speed_array), np.ravel(cmd_array))])
    acc_array = model.predict(grid_array).reshape(speed_array.shape)

    for cmd_index in range(cmd_segment_num):
        for speed_index in range(speed_segment_num):
            item = calibration_table_pb.calibration.add()
            item.speed = speed_array[cmd_index][speed_index]
            item.command = cmd_array[cmd_index][speed_index]
            item.acceleration = acc_array[cmd_index][speed_index]

    path = target_dir

    calibration_table_utils.write_h5_cal_tab(acc_array, path, table_filename)

    with open(os.path.join(path, table_filename), 'w') as wf:
        wf.write(str(calibration_table_pb))
    return os.path.join(path, table_filename)


def train_write_model(elem, target_prefix):
    (vehicle, throttle_or_brake), (data_set, train_param) = elem
    table_filename = throttle_or_brake + '_calibration_table.pb.txt'
    target_dir = os.path.join(target_prefix, vehicle)
    ((speed_min, speed_max, speed_segment_num),
     (cmd_min, cmd_max, cmd_segment_num), layer, train_alpha) = train_param
    model = train_model(data_set, layer, train_alpha)
    return (vehicle, write_table(model, target_dir, speed_min, speed_max, speed_segment_num,
                                 cmd_min, cmd_max, cmd_segment_num, table_filename))


def combine_file(files):
    brake_file, _ = files
    file_name = os.path.join(os.path.dirname(brake_file), 'calibration_table_pre.pb.txt')
    with open(file_name, 'wb') as outfile:
        for f in files:
            logging.info('infile: %s' % f)
            with open(f, "rb") as infile:
                for line in infile:
                    outfile.write(line)
    return file_name


def sort_single_config(single_file):
    """ sort pb.txt file w.r.t speed """
    calibration_table_pb = calibration_table_pb2.ControlCalibrationTable()
    origin_config = proto_utils.get_pb_from_text_file(single_file, calibration_table_pb)
    origin_config.calibration.sort(key=lambda elem: elem.speed)
    # write
    file_name = os.path.join(os.path.dirname(single_file), 'calibration_table.pb.txt')
    proto_utils.write_pb_to_text_file(origin_config, file_name)
