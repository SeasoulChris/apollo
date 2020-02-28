# offline_features_pb2


class LabelGenerator(object):
    def __init__(self):
        self.filepath = None
        '''
            feature_dict contains the organized Feature in the following way:
                obstacle_ID --> [Feature1, Feature2, Feature3, ...] (sequentially sorted)
            '''
        self.feature_dict = dict()

        def LoadFeaturePBAndSaveLabelFiles(self, input_filepath):
            '''
            This function will be used to replace all the functionalities in
            generate_cruise_label.py
            '''
            self.filepath = input_filepath
            offline_features = offline_features_pb2.Features()
            offline_features = proto_utils.get_pb_from_bin_file(self.filepath, offline_features)
            feature_sequences = offline_features.feature
            self.OrganizeFeatures(feature_sequences)
            del feature_sequences  # Try to free up some memory
            self.ObserveAllFeatureSequences()

        '''
        @brief: organize the features by obstacle IDs first, then sort each
                obstacle's feature according to time sequence.
        @input features: the unorganized features
        @output: organized (meaning: grouped by obstacle ID and sorted by time)
                features.
        '''

        def OrganizeFeatures(self, features):
            # Organize Feature by obstacle_ID (put those belonging to the same obstacle together)
            for feature in features:
                if feature.id in self.feature_dict.keys():
                    self.feature_dict[feature.id].append(feature)
                else:
                    self.feature_dict[feature.id] = [feature]

            # For the same obstacle, sort the Feature sequentially.
            for obs_id in self.feature_dict.keys():
                if len(self.feature_dict[obs_id]) < 2:
                    del self.feature_dict[obs_id]
                    continue
                self.feature_dict[obs_id].sort(key=lambda x: x.timestamp)

        '''
        @brief: observe all feature sequences and build observation_dict.
        @output: the complete observation_dict.
        '''

        def ObserveAllFeatureSequences(self):
            for obs_id, feature_sequence in self.feature_dict.items():
                for idx, feature in enumerate(feature_sequence):
                    if not feature.HasField('lane') or \
                            not feature.lane.HasField('lane_feature'):
                        # print('No lane feature, cancel labeling')
                        continue
                    self.ObserveFeatureSequence(feature_sequence, idx)
            np.save(self.filepath + '.npy', self.observation_dict)

        def ObserveFeatureSequence(self, feature_sequence, idx_curr):
            # Initialization.
            feature_curr = feature_sequence[idx_curr]
            dict_key = "{}@{:.3f}".format(feature_curr.id, feature_curr.timestamp)
            if dict_key in self.observation_dict.keys():
                return

        def LabelTrajectory(self, period_of_interest=3.0):
            output_features = offline_features_pb2.Features()
            for obs_id, feature_sequence in self.feature_dict.items():
                for idx, feature in enumerate(feature_sequence):
                    # Observe the subsequent Features
                    if "{}@{:.3f}".format(feature.id, feature.timestamp) not in self.observation_dict:
                        continue
                    observed_val = self.observation_dict["{}@{:.3f}".format(
                        feature.id, feature.timestamp)]
                    key = "{}@{:.3f}".format(feature.id, feature.timestamp)
                    self.future_status_dict[key] = observed_val['obs_traj']
            np.save(self.filepath + '.future_status.npy', self.future_status_dict)
