#!/usr/bin/env python
""" utils for multiple vehicles """
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common.proto_utils as proto_utils
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

from fueling.common.base_pipeline import BasePipeline


def get_vehicle(input_folder):
    return [folder.rsplit('/') for folder in os.listdir(input_folder)]


def get_vehicle_rdd(origin_prefix):
    # RDD(origin_dir)
    return (
        BasePipeline.context().parallelize([origin_prefix])
        # RDD([vehicle_type])
        .flatMap(get_vehicle)
        # PairRDD(vehicle_type, [vehicle_type])
        .keyBy(lambda vehicle_type: vehicle_type[0])
        # PairRDD(vehicle_type, path_to_vehicle_type)
        .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type[0])))


def get_conf(conf_dir):
    vehicle_conf_filename = 'vehicle_param.pb.txt'
    conf_file = os.path.join(conf_dir, vehicle_conf_filename)
    VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(
        conf_file, vehicle_config_pb2.VehicleConfig())
    return VEHICLE_PARAM_CONF.vehicle_param


DIM_INPUT = 3
input_index = {
    'speed': 0,  # chassis.speed_mps
    'acceleration': 1,
    'control command': 2,
}


def plot_feature_hist(elem):
    vehicle, feature = elem
    prefix = '/apollo/modules/data/fuel/testdata/control/generated/CalibrationTableFeature'
    timestr = time.strftime('%Y%m%d-%H%M%S')
    result_file = os.path.join(prefix, vehicle, 'Dataset_Distribution_%s.pdf' % timestr)
    with PdfPages(result_file) as pdf:
        for j in range(DIM_INPUT):
            plt.figure(figsize=(4, 3))
            plt.hist(feature[:, j], bins='auto')
            plt.title("Histogram of the " + list(input_index)[j])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
    return result_file
