#!/usr/bin/env python
""" Control feature visualization related utils """

import glob
import h5py

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np
import os

from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, FEATURE_NAMES
import fueling.profiling.conf.open_space_planner_conf as OpenSpaceConf
from fueling.profiling.proto.control_profiling_pb2 import ControlProfiling
from fueling.profiling.proto.control_profiling_data_pb2 import ControlFeatures
import fueling.common.h5_utils as h5_utils
import fueling.common.json_utils as json_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from fueling.profiling.conf.control_channel_conf import FEATURE_IDX, FEATURE_NAMES
from fueling.profiling.proto.control_profiling_data_pb2 import ControlFeatures
from fueling.profiling.proto.control_profiling_pb2 import ControlProfiling


def generate_segments(h5s):
    """generate data segments from all the selected hdf5 files"""
    segments = []
    if not h5s:
        logging.warning('No hdf5 files found under the targeted path.')
        return segments
    for h5 in h5s:
        logging.info('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as h5file:
            for value in h5file.values():
                segments.append(np.array(value))
    print('Segments width is: ', len(segments[0]))
    print('Segments length is: ', len(segments))
    return segments


def generate_data(segments):
    """generate data array from the given data segments"""
    data = []
    if not segments:
        logging.warning('No segments from hdf5 files found under the targetd path.')
        return data
    data.append(segments[0])
    for i in range(1, len(segments)):
        data = np.vstack([data, segments[i]])
    print('Data_Set length is: ', len(data))
    return data


def write_data_json_file(data_rdd):
    """ generate feature data and statistics json files for the given data array"""
    # PairRDD(target_dir, data_array)
    dir_data, data = data_rdd
    if len(data) == 0:
        logging.warning('No data from hdf5 files can be visualized under the targetd path {}'
                        .format(dir_data))
        return
    if data.shape[1] is not len(ControlFeatures().DESCRIPTOR.fields):
        logging.warning('The data size {} from hdf5 files does not match the proto field size {}'
                        .format(data.shape[1], len(ControlFeatures().DESCRIPTOR.fields)))
        return
    # define the control feature data file and feature statistics file names
    grading_dir = glob.glob(os.path.join(dir_data, '*grading.txt'))
    if grading_dir:
        json_data_file = os.path.basename(grading_dir[0]).replace(
            'control_performance_grading.txt', 'control_feature_data')
        json_statistics_file = os.path.basename(grading_dir[0]).replace(
            'control_performance_grading.txt', 'control_feature_statistics')
    else:
        json_data_file = 'control_feature_data'
        json_statistics_file = 'control_feature_statistics'
    # write the control feature data into the json files
    control_features = ControlFeatures()
    json_utils.get_pb_from_numpy_array(data, control_features)
    data_json = proto_utils.pb_to_dict(control_features)
    data_json['labels'] = {'x_label': 'timestamp_sec',
                           'y_label': [key for key in data_json.keys()
                                       if key is not 'timestamp_sec']}
    logging.info('transforming {} messages to json file {} for target {}'
                 .format(data.shape[0], json_data_file, dir_data))
    json_utils.write_json(data_json, dir_data, json_data_file)
    # write the control feature statistic values into the json files
    statistics_json = dict()
    for i in range(0, data.shape[1]):
        key = ControlFeatures().DESCRIPTOR.fields[i].name
        statistics_json[key] = {'mean': float('%.6f' % np.mean(data[:, i], axis=0)),
                                'standard deviation': float('%.6f' % np.std(data[:, i],
                                                                            axis=0))}
    json_utils.write_json(statistics_json, dir_data, json_statistics_file)


def clean_data(data, seq):
    """clean the data until the distribution matches the given standard"""
    length = data.shape[0]
    for i in range(9):
        idx_h = seq[int(length * (1 - 0.05 * i) - 1)]
        idx_l = seq[int(length * (0.05 * i))]
        scope = data[idx_h] - data[idx_l]
        idx_h_partial = seq[int(length * (1 - 0.05 * (i + 1)) - 1)]
        idx_l_partial = seq[int(length * (0.05 * (i + 1)))]
        scope_partial = data[idx_h_partial] - data[idx_l_partial]
        if (scope <= 2 * scope_partial):
            return [0.05 * i, 1 - 0.05 * i]
    return [0.05 * i, 1 - 0.05 * i]

    # TODO(Una/Vivian): This is only used in open space planner, and we may need to re-factor this out. 
def plot_hist(data_rdd):
    dir_data, group_id, data = data_rdd
    pdffile = os.path.join(dir_data, 'visualization.pdf')
    feature_names = OpenSpaceConf.FEATURE_NAMES
    with PdfPages(pdffile) as pdf:
        for i in range(len(feature_names)):
            data_plot = data[:, i]
            seq = np.argsort(data_plot)
            bounds = clean_data(data_plot, seq)
            logging.info('Processing the plots at Column: {}, Feature: {}'
                         .format(i, feature_names[i]))
            length = data_plot.shape[0]
            plt.figure(figsize=(4, 3))
            plt.hist(data_plot[seq[int(length * bounds[0]):int(length * bounds[1] - 1)]],
                     bins=100)
            plt.xlabel(feature_names[i])
            plt.ylabel('Sample length')
            plt.title("Histogram of " + feature_names[i] + " ("
                      + str(int(round((bounds[1] - bounds[0]) * 100))) + "% data)")
            xmin, xmax, ymin, ymax = plt.axis()
            plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                     'Maximum = {0:.3f}, Minimum = {1:.3f}'
                     .format(data_plot[seq[int(length * bounds[1] - 1)]],
                             data_plot[seq[int(length * bounds[0])]]),
                     color='red', fontsize=8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(4, 3))
            plt.plot(data_plot)
            plt.ylabel(feature_names[i])
            plt.xlabel('Sample Number')
            plt.title("Plot of " + feature_names[i] + " (100% Data)")
            xmin, xmax, ymin, ymax = plt.axis()
            plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                     'Maximum = {0:.3f}, Minimum = {1:.3f}'
                     .format(data_plot[seq[length - 1]], data_plot[seq[0]]),
                     color='red', fontsize=8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_h5_features_hist(data_rdd):
    """plot the histogram of all the variables in the data array"""
    # PairRDD(target_dir, data_array)
    dir_data, data = data_rdd
    if len(data) == 0:
        logging.warning('No data from hdf5 files can be visualized under the targeted path {}'
                        .format(dir_data))
        return
    grading_dir = glob.glob(os.path.join(dir_data, '*grading.txt'))
    if grading_dir:
        vehicle_controller = os.path.basename(grading_dir[0]).replace(
            'control_performance_grading.txt', '')
        pdffile = os.path.join(dir_data, vehicle_controller + 'control_data_visualization.pdf')
    else:
        pdffile = os.path.join(dir_data, 'control_data_visualization.pdf')

    profiling_conf = proto_utils.get_pb_from_text_file(
        '/fuel/fueling/profiling/conf/control_profiling_conf.pb.txt',
        ControlProfiling())

    with PdfPages(pdffile) as pdf:
        for i in range(len(FEATURE_NAMES)):
            if (i < data.shape[1] and
                    (i < FEATURE_IDX["timestamp_sec"] or i > FEATURE_IDX["trajectory_sequence_num"])):
                logging.info('Processing the plots at Column: {}, Feature: {}'
                             .format(i, FEATURE_NAMES[i]))
                if i == FEATURE_IDX["pose_heading_offset"]:
                    data_plot_idx = np.where((data[:, FEATURE_IDX["speed"]] >
                                              profiling_conf.control_metrics.speed_stop) &
                                             (data[:, FEATURE_IDX["curvature_reference"]] <
                                              profiling_conf.control_metrics.curvature_harsh_limit))[0]
                    data_plot = np.take(data[:, i], data_plot_idx, axis=0)
                else:
                    data_plot = data[:, i]
                length = data_plot.shape[0]
                if length > profiling_conf.min_sample_size:
                    seq = np.argsort(data_plot)
                    scope = data_plot[seq[length - 1]] - data_plot[seq[0]]
                    scope_90 = data_plot[seq[int(length * 0.95 - 1)]] - \
                        data_plot[seq[int(length * 0.05)]]
                    logging.info('The data scope is: {} the intermedia-90% data scope is: {}'
                                 .format(scope, scope_90))
                    bounds = clean_data(data_plot, seq)
                    if bounds[0] == 0 and bounds[1] == 1:
                        plt.figure(figsize=(4, 3))
                        plt.hist(data_plot, bins=100)
                        plt.xlabel(FEATURE_NAMES[i])
                        plt.ylabel('Sample length')
                        plt.title("Histogram of " + FEATURE_NAMES[i])
                        xmin, xmax, ymin, ymax = plt.axis()
                        plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                                 'Maximum = {0:.3f}, Minimum = {1:.3f}'
                                 .format(data_plot[seq[length - 1]], data_plot[seq[0]]),
                                 color='red', fontsize=8)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                    else:
                        plt.figure(figsize=(4, 3))
                        plt.hist(data_plot[seq[int(length * bounds[0]):int(length * bounds[1] - 1)]],
                                 bins=100)
                        plt.xlabel(FEATURE_NAMES[i])
                        plt.ylabel('Sample length')
                        plt.title("Histogram of " + FEATURE_NAMES[i] + " ("
                                  + str(int(round((bounds[1] - bounds[0]) * 100))) + "% data)")
                        xmin, xmax, ymin, ymax = plt.axis()
                        plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                                 'Maximum = {0:.3f}, Minimum = {1:.3f}'
                                 .format(data_plot[seq[int(length * bounds[1] - 1)]],
                                         data_plot[seq[int(length * bounds[0])]]),
                                 color='red', fontsize=8)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                        plt.figure(figsize=(4, 3))
                        plt.plot(data_plot)
                        plt.ylabel(FEATURE_NAMES[i])
                        plt.xlabel('Sample Number')
                        plt.title("Plot of " + FEATURE_NAMES[i] + " (100% Data)")
                        xmin, xmax, ymin, ymax = plt.axis()
                        plt.text(xmin * 0.9 + xmax * 0.1, ymin * 0.1 + ymax * 0.9,
                                 'Maximum = {0:.3f}, Minimum = {1:.3f}'
                                 .format(data_plot[seq[length - 1]], data_plot[seq[0]]),
                                 color='red', fontsize=8)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
