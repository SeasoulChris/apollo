#!/usr/bin/env python

import argparse
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
        self.secondary_filepath = None
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
        self.label_dict = dict()
        # planning tag dict (key: timestamp, val: planning_tag)
        self.planning_tag_dict = dict()
        # history dict for debug purpose
        self.history_adc_trajectory_dict = dict()

    def LoadPBFiles(self, input_filepath, output_filepath, secondary_filepath=None):
        self.src_filepath = input_filepath
        logging.info(input_filepath)

        self.dst_filepath = output_filepath
        logging.info(output_filepath)
        # load origin PB file
        offline_features = proto_utils.get_pb_from_bin_file(
            self.src_filepath, learning_data_pb2.LearningData())
        # get learning data sequence
        learning_data_sequence = offline_features.learning_data
        origin_data_len = len(learning_data_sequence)  # origin data length
        logging.info(len(learning_data_sequence))

        if secondary_filepath:
            self.secondary_filepath = secondary_filepath
            # extra trajectory
            extra_offline_features = proto_utils.get_pb_from_bin_file(
                self.secondary_filepath, offline_features)
            # learning_data from current bin + learning data from next bin
            learning_data_sequence.extend(extra_offline_features.learning_data)
            logging.info(len(learning_data_sequence))
        return learning_data_sequence, origin_data_len

    def UpdatePlanningTag(self, dict_key, current_tag, future_tag):
        # campare one trajectory point in different time frame
        # update trajectory point overlapFeature field only when this field is not setted
        # to guaranttee the nearest OverlapFeature
        tag_bp = learning_data_pb2.PlanningTag()
        current_tag_dict = proto_utils.pb_to_dict(current_tag)
        future_tag_dict = proto_utils.pb_to_dict(future_tag)
        # find the unsetted overlap
        unsetted_tag = set(future_tag_dict.keys()) - set(current_tag_dict.keys())
        for key in unsetted_tag:
            current_tag_dict[key] = future_tag_dict[key]
        updated_tag = proto_utils.dict_to_pb(current_tag_dict, tag_bp)
        logging.debug(f'*****{dict_key}: current_tag_dict:{current_tag_dict}')
        logging.debug(f'future_tag_dict:{future_tag_dict}')
        return updated_tag

    def GetObserveAllFeatureSequences(
            self, input_filepath, output_filepath, secondary_filepath=None):
        learning_data_sequence, origin_data_len = self.LoadPBFiles(
            input_filepath, output_filepath, secondary_filepath)
        # get all trajectory points from feature_sequence
        adc_trajectory = []
        timestamps = []
        for idx, learning_data in enumerate(learning_data_sequence):
            # key + feature
            for adc_trajectory_point in learning_data.adc_trajectory_point:
                # for adc_trajectory_point in reversed(learning_data.adc_trajectory_point):
                # assuming last point is the lastest/newest/current trajectory point
                # 1. write a newer trajectory point to trajectory point tag list
                dict_key = "adc@{:.3f}".format(adc_trajectory_point.timestamp_sec)
                logging.debug(f'dict_key: {dict_key}')

                # add planning tag
                if dict_key in self.planning_tag_dict:
                    # decide whether to update current planning_tag
                    self.planning_tag_dict[dict_key] = self.UpdatePlanningTag(
                        dict_key, self.planning_tag_dict[dict_key], adc_trajectory_point.planning_tag)
                else:
                    # first time encounter
                    self.planning_tag_dict[dict_key] = adc_trajectory_point.planning_tag

                # add trajectory point
                if adc_trajectory_point.timestamp_sec not in timestamps:
                    timestamps.append(adc_trajectory_point.timestamp_sec)
                    adc_trajectory.append(adc_trajectory_point)
            # last trajectory point has the latest timestamp
            frame_key = "adc@{:.3f}".format(timestamps[-1])
            logging.debug(f'No {idx} frame_key: {frame_key}')
            # key: current localization point
            if idx < origin_data_len:  # first part of the list is from origin PB file
                # key of each learning_data is the timestamps of current trajectory point
                self.feature_dict[frame_key] = learning_data

        # [Feature1, Feature2, Feature3, ...] (sequentially sorted)
        adc_trajectory.sort(key=lambda x: x.timestamp_sec)
        logging.info(len(adc_trajectory))
        self.feature_sequence = adc_trajectory
        self.ObserveAllFeatureSequences()

    def WriteTagToFrame(self, is_dump2bin=False, is_dump2txt=False):
        features_tags = learning_data_pb2.LearningData()
        learning_data_frame = learning_data_pb2.LearningDataFrame()
        for key in self.feature_dict.keys():
            # write feature to proto
            learning_data_frame = self.feature_dict[key]
            # write tag to proto when tag exists
            if key in self.planning_tag_dict:
                learning_data_frame.planning_tag.CopyFrom(self.planning_tag_dict[key])
                # write planning tag to feature_dict
                self.feature_dict[key] = learning_data_frame
            features_tags.learning_data.add().CopyFrom(learning_data_frame)
        # export proto to bin
        if is_dump2bin:
            with open(self.dst_filepath + '.with_tag.bin', 'wb') as bin_f:
                bin_f.write(features_tags.SerializeToString())
        if is_dump2txt:
            # export proto to txt
            txt_file_name = self.dst_filepath + '.with_tag.txt'
            # export single frame to txt for debug
            proto_utils.write_pb_to_text_file(features_tags.learning_data[0], txt_file_name)
        return len(features_tags.learning_data)

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

    def ObserveFeatureSequence(self, feature_sequence, idx_curr, maximum_observation_range=60):
        output_features = learning_data_pb2.LearningOutput()
        # Initialization.
        feature_curr = feature_sequence[idx_curr]
        dict_key = "adc@{:.3f}".format(feature_curr.timestamp_sec)
        if dict_key in self.observation_dict.keys():
            return
        # Declare needed varables.
        is_jittering = False
        feature_seq_len = len(feature_sequence)
        adc_traj = []
        total_observed_time_span = 0.0
        # fix data point numbers
        future_start_index = idx_curr + 1

        # This goes through all the subsequent features in this sequence
        # of features up to the maximum_observation_time.
        for j in range(future_start_index,
                       min(future_start_index + maximum_observation_range, len(feature_sequence))):
            # logging.info(feature_sequence[j].timestamp_sec)
            # timestamp_sec: 0.0
            # trajectory_point {
            #   path_point {
            #     x: 587027.5898016331
            #     y: 4140999.7741826824
            #     z: 0.0
            #     theta: -0.2452333360636869
            #   }
            #   v: 2.912844448832799
            #   a: 0.0029292981829968997
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
            total_observed_time_span = feature_sequence[j].timestamp_sec - \
                feature_curr.timestamp_sec

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
            logging.debug(len(observed_val[1]['adc_traj']))
            time_span = observed_val[1]['total_observed_time_span']
            logging.debug(time_span)
        logging.info(self.dst_filepath)
        logging.info("dst file: {}".format(self.dst_filepath + '.future_status.npy'))
        np.save(self.dst_filepath + '.future_status.npy', self.future_status_dict)
        return self.future_status_dict

    def GetHistoryTrajectory(self):
        """ for debug purpose """
        for key in self.feature_dict:
            current_learning_data = self.feature_dict[key]
            # get history trajectory from each learning frame
            adc_traj = []
            for history_adc_trajectory_point in current_learning_data.adc_trajectory_point:
                adc_traj.append((history_adc_trajectory_point.trajectory_point.path_point.x,
                                 history_adc_trajectory_point.trajectory_point.path_point.y,
                                 history_adc_trajectory_point.trajectory_point.path_point.z,
                                 history_adc_trajectory_point.trajectory_point.path_point.theta,
                                 history_adc_trajectory_point.trajectory_point.v,
                                 history_adc_trajectory_point.trajectory_point.a,
                                 history_adc_trajectory_point.timestamp_sec))
            self.history_adc_trajectory_dict[key] = adc_traj
        logging.info(f'history path: {self.dst_filepath }.history_status.npy')
        np.save(self.dst_filepath + '.history_status.npy', self.history_adc_trajectory_dict)
        return self.history_adc_trajectory_dict

    def MergeDict(self, is_dump2txt=True):
        """ merge feature and label """
        features_labels = learning_data_pb2.LearningData()
        learning_data_frame = learning_data_pb2.LearningDataFrame()
        for key in self.label_dict.keys():
            # write feature to proto
            learning_data_frame = self.feature_dict[key]
            # write label to proto
            learning_data_frame.output.CopyFrom(self.label_dict[key])
            features_labels.learning_data.add().CopyFrom(learning_data_frame)
            self.feature_label_dict[key] = (
                self.label_dict[key], self.feature_dict[key])
        # export proto to bin
        with open(self.dst_filepath + '.future_status.bin', 'wb') as bin_f:
            bin_f.write(features_labels.SerializeToString())
        if is_dump2txt:
            # export proto to txt
            # exprot single frame for debug
            txt_file_name = self.dst_filepath + '.future_status.txt'
            proto_utils.write_pb_to_text_file(features_labels.learning_data[0], txt_file_name)
        return len(features_labels.learning_data)

    def Label(self):
        # add tag to data frame
        self.WriteTagToFrame()
        # generate label
        self.LabelTrajectory()
        # add label to data frame
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
    parser = argparse.ArgumentParser(description='labeling')
    parser.add_argument(
        '--input_file', type=str,
        default='/apollo/data/learning_based_planning/input/00036.record.6.bin')
    parser.add_argument(
        '--secondary_input_file', type=str)

    # output file name is modified in code
    parser.add_argument(
        '--output_file', type=str,
        default='/apollo/data/learning_based_planning/output/labeled_data.bin')
    parser.add_argument(
        '--future_img_output_file', type=str,
        default='/apollo/data/learning_based_planning/output/future_trajectory.pdf')
    parser.add_argument(
        '--history_img_output_file', type=str,
        default='/apollo/data/learning_based_planning/output/history_trajectory.pdf')
    parser.add_argument(
        '--key_id', type=int,
        default='20')

    args = parser.parse_args()
    label_gen = LabelGenerator()
    if args.secondary_input_file:
        result = label_gen.GetObserveAllFeatureSequences(
            args.input_file, args.output_file, args.secondary_input_file)
    else:
        result = label_gen.GetObserveAllFeatureSequences(
            args.input_file, args.output_file)
    label_gen.WriteTagToFrame()  # write planning tag to learning data
    result2 = label_gen.LabelTrajectory()
    logging.info(len(result2))
    history_result2 = label_gen.GetHistoryTrajectory()
    key_list = list(history_result2.keys())
    history_data_points = history_result2[key_list[args.key_id]]
    label_gen.Visualize(history_data_points, args.history_img_output_file)
    data_points = result2[key_list[args.key_id]]
    # logging.info(data_points)
    label_gen.Visualize(data_points, args.future_img_output_file)
    label_gen.Label()
