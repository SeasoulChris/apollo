#!/usr/bin/env python
import glob
import os

import numpy as np
import pyspark_utils.op as spark_op

from modules.prediction.proto import offline_features_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class FeaturesAndLabelsCombine(BasePipeline):
    """Records to DataForLearning proto pipeline."""

    def run_test(self):
        """Run test."""
        datalearn_files = self.to_rdd(glob.glob('/apollo/docs/demo_guide/*/datalearn.*.bin'))
        self.run(datalearn_files)

    def run_prod(self):
        """Run prod."""
        origin_prefix = 'modules/prediction/features-san-mateo'

        # RDD(datalearn_file)
        datalearn_file_rdd = self.to_rdd(self.our_storage().list_files(origin_prefix)).filter(
            spark_op.filter_path(['*datalearn.*.bin']))

        self.run(datalearn_file_rdd)

    def run(self, datalearn_file_rdd):
        """Run the pipeline with given arguments."""
        # RDD(0/1), 1 for success
        result = datalearn_file_rdd.map(self.process_dir).count()
        logging.info('Processed {} tasks'.format(result))

    @staticmethod
    def process_dir(source_file):
        """Call prediction python code to combine features and labels"""
        source_dir = os.path.dirname(source_file)
        labels_dir = source_dir.replace('features-san-mateo', 'labels-san-mateo')
        label_file = os.path.join(labels_dir, 'junction_label.npy')
        CombineFeaturesAndLabels(source_file, label_file, 'junction_label')
        return 0


def CombineFeaturesAndLabels(feature_path, label_path, dict_name='future_status'):
    list_of_data_for_learning = offline_features_pb2.ListDataForLearning()
    with open(feature_path, 'rb') as file_in:
        list_of_data_for_learning.ParseFromString(file_in.read())
    dict_labels = np.load(label_path).item()

    junction_label_label_dim = 24
    future_status_label_dim = 30

    output_np_array = []
    for data_for_learning in list_of_data_for_learning.data_for_learning:
        # features_for_learning: list of doubles
        features_for_learning = list(data_for_learning.features_for_learning)
        key = "{}@{:.3f}".format(data_for_learning.id, data_for_learning.timestamp)

        # Sanity checks to see if this data-point is valid or not.
        if key not in dict_labels:
            logging.info('Cannot find a feature-to-label mapping.')
            continue

        labels = None
        list_curr = None
        if dict_name == 'junction_label':
            if len(dict_labels[key]) != junction_label_label_dim:
                continue
            labels = dict_labels[key]
            list_curr = features_for_learning + labels
        elif dict_name == 'future_status':
            if len(dict_labels[key]) < future_status_label_dim:
                continue
            labels = dict_labels[key][:future_status_label_dim]
            list_curr = [len(features_for_learning)] + features_for_learning + labels
        elif dict_name == 'visited_lane_segment':
            labels = dict_labels[key]
            list_curr = [len(features_for_learning)] + features_for_learning + \
                        [len(string_features_for_learning)] + string_features_for_learning + \
                labels + [len(labels)]
        elif dict_name == 'cruise_label':
            lane_seq_id = data_for_learning.lane_sequence_id
            if lane_seq_id not in dict_labels[key]:
                continue
            labels = dict_labels[key][lane_seq_id]
            lane_seq_label = labels[0]
            list_curr = features_for_learning + [lane_seq_id]

        if list_curr is None:
            continue
        output_np_array.append(list_curr)

    output_np_array = np.array(output_np_array)

    np.save(feature_path + '.features+' + dict_name + '.npy', output_np_array)
    return output_np_array


if __name__ == '__main__':
    FeaturesAndLabelsCombine().main()
