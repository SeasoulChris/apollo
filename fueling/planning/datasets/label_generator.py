#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt

from modules.planning.proto import learning_data_pb2
import fueling.common.proto_utils as proto_utils
import fueling.common.logging as logging

from fueling.planning.data_visualizer import mkz_plotter


class LabelGenerator(object):
    def __init__(self):
        self.src_filepath = None
        self.dst_filepath = None
        self.feature_sequence = []
        # super set of label
        self.observation_dict = dict()
        # label: future adc trajectory
        self.future_status_dict = dict()
        # feature: history info
        self.feature_dict = dict()
        # training data with label
        self.feature_label_dict = dict()
        # label dict
        self. label_dict = dict()

    def LoadFeaturePBAndSaveLabelFiles(self, input_filepath, output_filepath):
        self.src_filepath = input_filepath
        logging.info(input_filepath)
        self.dst_filepath = output_filepath
        logging.info(output_filepath)
        offline_features = learning_data_pb2.LearningData()
        offline_features = proto_utils.get_pb_from_bin_file(self.src_filepath, offline_features)
        learning_data_sequence = offline_features.learning_data
        # get all trajectory points from feature_sequence
        adc_trajectory = []
        timestamps = []
        # logging.info(learning_data_sequence[-1])
        logging.info(len(offline_features.learning_data))
        for idx, learning_data in enumerate(learning_data_sequence):
            # key + feature
            # logging.info(learning_data.adc_trajectory_point)
            for adc_trajectory_point in learning_data.adc_trajectory_point:
                # remove duplication
                if adc_trajectory_point.timestamp_sec not in timestamps:
                    timestamps.append(adc_trajectory_point.timestamp_sec)
                    adc_trajectory.append(adc_trajectory_point)
            # key: current localization point
            self.feature_dict["adc@{:.3f}".format(timestamps[-1])] = learning_data
        # print(adc_trajectory[-1])
        # [Feature1, Feature2, Feature3, ...] (sequentially sorted)
        adc_trajectory.sort(key=lambda x: x.timestamp_sec)
        logging.info(len(adc_trajectory))
        # logging.info(adc_trajectory[-1].timestamp_sec)
        self.feature_sequence = adc_trajectory
        return self.ObserveAllFeatureSequences()

    '''
    @brief: observe all feature sequences and build observation_dict.
    @output: the complete observation_dict.
    '''

    def ObserveAllFeatureSequences(self):
        for idx, feature in enumerate(self.feature_sequence):
            self.ObserveFeatureSequence(self.feature_sequence, idx)
        # np.save(self.src_filepath + '.npy', self.observation_dict)
        logging.info(len(self.feature_dict))
        logging.debug(self.feature_dict.keys())
        return

    '''
    @brief: Observe the sequence of Features following the Feature at
            idx_curr and save some important observations in the class.
    @input feature_sequence: A sorted sequence of Feature corresponding to adc.
    @input idx_curr: The index of the current Feature to be labelled.
                     We will look at the subsequent Features following this
                     one to complete labeling.
    @output: All saved as class variables in observation_dict,
    '''

    def ObserveFeatureSequence(self, feature_sequence, idx_curr):
        output_features = learning_data_pb2.LearningOutput()
        # Initialization.
        feature_curr = feature_sequence[idx_curr]
        dict_key = "adc@{:.3f}".format(feature_curr.timestamp_sec)
        if dict_key in self.observation_dict.keys():
            return
        # Declare needed varables.
        is_jittering = False
        feature_seq_len = len(feature_sequence)
        prev_timestamp = -1.0
        adc_traj = []
        total_observed_time_span = 0.0
        maximum_observation_time = 3.0

        # This goes through all the subsequent features in this sequence
        # of features up to the maximum_observation_time.
        for j in range(idx_curr, feature_seq_len):
            # logging.info(feature_sequence[j].timestamp_sec )
            # If timespan exceeds max. observation time, then end observing.
            time_span = feature_sequence[j].timestamp_sec - feature_curr.timestamp_sec
            if time_span > maximum_observation_time:
                break
            total_observed_time_span = time_span
            # timestamp_sec: 0.0
            # trajectory_point {
            #   path_point {
            #     x: 587027.5898016331
            #     y: 4140950.7741826824
            #     z: 0.0
            #     theta: -0.2452333360636869
            #   }
            #   v: 2.912844448832799
            #   a: 0.0029292981825068507
            # }
            #####################################################################
            # Update the ADC trajectory:
            # Only update for consecutive (sampling rate = 0.1sec) points.
            adc_traj.append((feature_sequence[j].trajectory_point.path_point.x,
                             feature_sequence[j].trajectory_point.path_point.y,
                             feature_sequence[j].trajectory_point.path_point.z,
                             feature_sequence[j].trajectory_point.path_point.theta,
                             feature_sequence[j].trajectory_point.v,
                             feature_sequence[j].trajectory_point.a,
                             feature_sequence[j].timestamp_sec))
            # proto form
            output_features.adc_future_trajectory_point.add().CopyFrom(
                feature_sequence[j])
        # logging.info(len(output_features.adc_future_trajectory_point))
        # for adc_future_trajectory_point in output_features.adc_future_trajectory_point:
        #     logging.info(adc_future_trajectory_point.timestamp_sec)
        # Update the observation_dict:
        dict_val = dict()
        dict_val['adc_traj'] = adc_traj
        dict_val['adc_traj_len'] = len(adc_traj)
        dict_val['is_jittering'] = is_jittering
        dict_val['total_observed_time_span'] = total_observed_time_span
        key = "adc@{:.3f}".format(feature_curr.timestamp_sec)
        self.observation_dict[key] = ((output_features), (dict_val))
        return

    def LabelTrajectory(self):
        for idx, feature in enumerate(self.feature_sequence):
            # Observe the subsequent Features
            if "adc@{:.3f}".format(feature.timestamp_sec) not in self.feature_dict:
                continue
            observed_val = self.observation_dict["adc@{:.3f}".format(feature.timestamp_sec)]
            key = "adc@{:.3f}".format(feature.timestamp_sec)
            self.label_dict[key] = observed_val[0]  # output_features
            self.future_status_dict[key] = observed_val[1]['adc_traj']
        logging.info(self.dst_filepath)
        logging.info("dst file: {}".format(self.dst_filepath + '.future_status.npy'))
        np.save(self.dst_filepath + '.future_status.npy', self.future_status_dict)
        return self.future_status_dict

    def MergeDict(self):
        """ merge feature and label """
        features_labels = learning_data_pb2.LearningData()
        learning_data_frame = learning_data_pb2.LearningDataFrame()
        for key in self.label_dict.keys():
            # write feature to proto
            learning_data_frame.CopyFrom(self.feature_dict[key])
            # write label to proto
            learning_data_frame.output.CopyFrom(self.label_dict[key])
            features_labels.learning_data.add().CopyFrom(learning_data_frame)
            self.feature_label_dict[key] = (
                self.label_dict[key], self.feature_dict[key])
        # export proto to bin
        with open(self.dst_filepath + '.future_status.bin', 'wb') as bin_f:
            bin_f.write(features_labels.SerializeToString())
        return len(features_labels.learning_data)

    def Label(self):
        self.LabelTrajectory()
        self.MergeDict()

    def Visualize(self, data_points, img_fn):
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        ncolor = len(data_points)
        colorVals = plt.cm.jet(np.linspace(0, 1, ncolor))
        for idx, feature in enumerate(data_points):
            c = colorVals[idx]
            self.PlotAgent(feature, ax, c)
        plt.axis('equal')
        fig.savefig(img_fn)
        plt.close(fig)

    def PlotAgent(self, feature, ax, c):
        heading = feature[3]
        position = []
        position.append(feature[0])
        position.append(feature[1])
        position.append(feature[2])
        mkz_plotter.plot(position, heading, ax, c)


if __name__ == '__main__':
    FILE = '/apollo/data/learning_based_planning/bin_result/learning_data.0.bin'
    OUTPUT_FILE = '/apollo/data/learning_based_planning/npy_result/learning_data.0.bin'
    label_gen = LabelGenerator()
    result = label_gen.LoadFeaturePBAndSaveLabelFiles(FILE, OUTPUT_FILE)
    result2 = label_gen.LabelTrajectory()
    logging.info(len(result2))
    logging.debug(label_gen.MergeDict())
    data_points = result2['adc@1571344606.693']
    logging.debug(data_points)
    IMG_FN = '/apollo/data/learning_based_planning/learning_data.0.bin.pdf'
    label_gen.Visualize(data_points, IMG_FN)
    OUTPUT_BIN_FILE = '/apollo/data/learning_based_planning/npy_result/learning_data.0.bin.future_status.bin'
    offline_features = learning_data_pb2.LearningData()
    offline_features = proto_utils.get_pb_from_bin_file(OUTPUT_BIN_FILE, offline_features)
