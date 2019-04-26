#!/usr/bin/env python
""" utils for multiple vehicles """
import os
import time

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import colored_glog as glog
import matplotlib.pyplot as plt
import numpy as np

import common.proto_utils as proto_utils
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.s3_utils as s3_utils
import modules.data.fuel.fueling.control.proto.calibration_table_pb2 as CalibrationTable


FILENAME_CALIBRATION_TABLE_CONF = \
    '/apollo/modules/data/fuel/fueling/control/conf/calibration_table_conf.pb.txt'
CALIBRATION_TABLE_CONF = proto_utils.get_pb_from_text_file(FILENAME_CALIBRATION_TABLE_CONF,
                                                           CalibrationTable.CalibrationTable())

throttle_train_layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
                        CALIBRATION_TABLE_CONF.throttle_train_layer2,
                        CALIBRATION_TABLE_CONF.throttle_train_layer3]

brake_train_layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
                     CALIBRATION_TABLE_CONF.brake_train_layer2,
                     CALIBRATION_TABLE_CONF.brake_train_layer3]

train_alpha = CALIBRATION_TABLE_CONF.train_alpha


# def get_vehicle_rdd(origin_prefix):
#     # RDD(origin_dir)
#     return (
#         BasePipeline.context().parallelize([origin_prefix])
#         # RDD([vehicle_type])
#         .flatMap(get_vehicle)
#         # PairRDD(vehicle_type, [vehicle_type])
#         .keyBy(lambda vehicle_type: vehicle_type[0])
#         # PairRDD(vehicle_type, path_to_vehicle_type)
#         .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type[0])))


def get_vehicle_param_prod(prefix):
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    bucket = 'apollo-platform'
    return(
        s3_utils.list_files(bucket, prefix, vehicle_para_conf_filename)
        # PairRDD(vehicle, conf_file_path)
        .keyBy(lambda path: path.split('/')[-2])
        # PairRDD(vehicle, conf)
        .mapValues(lambda conf_file: proto_utils.get_pb_from_text_file(
            conf_file, vehicle_config_pb2.VehicleConfig()))
        # PairRDD(vehicle, vehicle_param)
        .mapValues(lambda vehicle_conf: vehicle_conf.vehicle_param))


# def get_conf(conf_dir):
#     vehicle_conf_filename = 'vehicle_param.pb.txt'
#     conf_file = os.path.join(conf_dir, vehicle_conf_filename)
#     VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(
#         conf_file, vehicle_config_pb2.VehicleConfig())
#     return VEHICLE_PARAM_CONF.vehicle_param


DIM_INPUT = 3
input_index = {
    0: 'speed',  # chassis.speed_mps
    1: 'acceleration',
    2: 'control command',
}


def plot_feature_hist(elem, target_dir):
    vehicle, feature = elem
    timestr = time.strftime('%Y%m%d-%H%M%S')
    result_file = os.path.join(target_dir, vehicle, 'Dataset_Distribution_%s.pdf' % timestr)
    with PdfPages(result_file) as pdf:
        for j in range(DIM_INPUT):
            plt.figure(figsize=(4, 3))
            plt.hist(feature[:, j], bins='auto')
            plt.title("Histogram of the " + input_index[j])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
    return result_file


# def gen_param(vehicle_param, throttle_or_brake):
#     if throttle_or_brake == 'throttle':
#         cmd_min = vehicle_param.throttle_deadzone
#         cmd_max = CALIBRATION_TABLE_CONF.throttle_max
#         layer = [CALIBRATION_TABLE_CONF.throttle_train_layer1,
#                  CALIBRATION_TABLE_CONF.throttle_train_layer2,
#                  CALIBRATION_TABLE_CONF.throttle_train_layer3]

#     elif throttle_or_brake == 'brake':
#         cmd_min = -1 * CALIBRATION_TABLE_CONF.brake_max
#         cmd_max = -1 * vehicle_param.brake_deadzone
#         layer = [CALIBRATION_TABLE_CONF.brake_train_layer1,
#                  CALIBRATION_TABLE_CONF.brake_train_layer2,
#                  CALIBRATION_TABLE_CONF.brake_train_layer3]

#     speed_min = CALIBRATION_TABLE_CONF.train_speed_min
#     speed_max = CALIBRATION_TABLE_CONF.train_speed_max
#     speed_segment_num = CALIBRATION_TABLE_CONF.train_speed_segment
#     cmd_segment_num = CALIBRATION_TABLE_CONF.train_cmd_segment
#     train_alpha = CALIBRATION_TABLE_CONF.train_alpha
#     return ((speed_min, speed_max, speed_segment_num),
#             (cmd_min, cmd_max, cmd_segment_num), layer, train_alpha)


def gen_plot(elem, target_dir, throttle_or_brake):

    (vehicle, (((speed_min, speed_max, speed_segment_num),
                (cmd_min, cmd_max, cmd_segment_num), layer, train_alpha), acc_maxtrix)) = elem

    timestr = time.strftime('%Y%m%d-%H%M%S')
    result_file = os.path.join(
        target_dir, vehicle, (throttle_or_brake + '_result_%s.pdf' % timestr))

    cmd_array = np.linspace(cmd_min, cmd_max, num=cmd_segment_num)
    speed_array = np.linspace(speed_min, speed_max, num=speed_segment_num)
    speed_maxtrix, cmd_matrix = np.meshgrid(speed_array, cmd_array)
    grid_array = np.array([[s, c] for s, c in zip(np.ravel(speed_array), np.ravel(cmd_array))])
    with PdfPages(result_file) as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(speed_maxtrix, cmd_matrix, acc_maxtrix,
                        alpha=1, rstride=1, cstride=1, linewidth=0.5, antialiased=True)
        ax.set_xlabel('$speed$')
        ax.set_ylabel('$throttle$')
        ax.set_zlabel('$acceleration$')
        pdf.savefig()
    return result_file
