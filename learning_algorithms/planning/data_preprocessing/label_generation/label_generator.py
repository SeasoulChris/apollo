#!/usr/bin/env python


import numpy as np

from modules.planning.proto import learning_data_pb2
import fueling.common.proto_utils as proto_utils


class LabelGenerator(object):
    def __init__(self):
        self.filepath = None
        '''
        observation_dict contains the important observations of the subsequent
        Features for each obstacle at every timestamp:
            obstacle_ID@timestamp --> dictionary of observations
        where dictionary of observations contains:
            'adu_traj': the trajectory points (x, y, vel_heading) up to
                        max_observation_time this trajectory point must
                        be consecutive (0.1sec sampling period)
            'adu_traj_len': length of trajectory points
            'is_jittering':
            'total_observed_time_span':
        This observation_dict, once constructed, can be reused by various labeling
        functions.
        '''
        self.observation_dict = dict()

    def LoadFeaturePBAndSaveLabelFiles(self, input_filepath):
        self.filepath = input_filepath
        offline_features = learning_data_pb2.LearningData()
        offline_features = proto_utils.get_pb_from_bin_file(self.filepath, offline_features)
        feature_sequence = offline_features.learning_data
        # [Feature1, Feature2, Feature3, ...] (sequentially sorted)
        feature_sequence.sort(key=lambda x: x.timestamp_sec)
        # return feature_sequences[-1].timestamp_sec
        return self.ObserveFeatureSequence(feature_sequence, 0)

    '''
    @brief: observe all feature sequences and build observation_dict.
    @output: the complete observation_dict.
    '''

    def ObserveAllFeatureSequences(self, feature_sequence):
        for idx, feature in enumerate(feature_sequence):
            self.ObserveFeatureSequence(feature_sequence)
        np.save(self.filepath + '.npy', self.observation_dict)

    def ObserveFeatureSequence(self, feature_sequence, idx_curr):
        # Initialization.
        feature_curr = feature_sequence[idx_curr]
        dict_key = "ADU@{:.3f}".format(feature_curr.timestamp_sec)
        if dict_key in self.observation_dict.keys():
            return
        # Declare needed varables.
        is_jittering = False
        feature_seq_len = len(feature_sequence)
        prev_timestamp = -1.0
        adu_traj = []
        total_observed_time_span = 0.0
        maximum_observation_time = 8.0

        # This goes through all the subsequent features in this sequence
        # of features up to the maximum_observation_time.
        for j in range(idx_curr, feature_seq_len):
            # If timespan exceeds max. observation time, then end observing.
            time_span = feature_sequence[j].timestamp_sec - feature_curr.timestamp_sec
            if time_span > maximum_observation_time:
                break
            total_observed_time_span = time_span

            #####################################################################
            # Update the ADU trajectory:
            # Only update for consecutive (sampling rate = 0.1sec) points.
            # IMPORTANT NOTE: APPEND ONLY to add new items
            adu_traj.append((feature_sequence[j].localization.position.x,
                             feature_sequence[j].localization.position.y,
                             feature_sequence[j].localization.heading,
                             feature_sequence[j].localization.linear_velocity.x,
                             feature_sequence[j].localization.linear_velocity.y,
                             feature_sequence[j].localization.linear_acceleration.x,
                             feature_sequence[j].localization.linear_acceleration.y,
                             feature_sequence[j].localization.angular_velocity.z,
                             feature_sequence[j].timestamp_sec))
        # Update the observation_dict:
        dict_val = dict()
        dict_val['adu_traj'] = adu_traj
        dict_val['adu_traj_len'] = len(adu_traj)
        dict_val['is_jittering'] = is_jittering
        dict_val['total_observed_time_span'] = total_observed_time_span
        self.observation_dict["@{:.3f}".format(feature_curr.timestamp_sec)] = dict_val
        np.save(self.filepath + '.npy', self.observation_dict)
        return dict_val

    # def LabelTrajectory(self, period_of_interest=3.0):
    #     output_features = learning_data_pb2.Features()
    #     for feature_sequence in self.feature_dict.items():
    #         for idx, feature in enumerate(feature_sequence):
    #             # Observe the subsequent Features
    #             if "{}@{:.3f}".format(feature.id, feature.timestamp) not in self.observation_dict:
    #                 continue
    #             observed_val = self.observation_dict["{}@{:.3f}".format(
    #                 feature.id, feature.timestamp)]
    #             key = "{}@{:.3f}".format(feature.id, feature.timestamp)
    #             self.future_status_dict[key] = observed_val['obs_traj']
    #     np.save(self.filepath + '.future_status.npy', self.future_status_dict)


# demo
if __name__ == '__main__':
    FILE = '/apollo/data/learning_based_planning/bin_result/learning_data.2.bin'
    label_gen = LabelGenerator()
    result = label_gen.LoadFeaturePBAndSaveLabelFiles(FILE)
    print(result)
