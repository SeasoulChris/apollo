#!/usr/bin/env python
import fnmatch
import glob
import operator
import os

import numpy as np
import pyspark_utils.op as spark_op

from modules.prediction.proto import offline_features_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class FeaturesAndLabelsCombine(BasePipeline):
    """Records to DataForLearning proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'combine-features-and-labels')

    def run_test(self):
        """Run test."""
        sc = self.get_spark_context()
        root_dir = '/apollo'
        datalearn_files = sc.parallelize(glob.glob('docs/demo_guide/*/datalearn.*.bin'))
        origin_prefix = 'docs/demo_guide'
        self.run(root_dir, datalearn_files, origin_prefix)

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        bucket = 'apollo-platform'
        origin_prefix = 'modules/prediction/features-san-mateo/'

        datalearn_file_rdd = (
            # RDD(file), start with origin_prefix
            s3_utils.list_files(bucket, origin_prefix)
            # RDD(datalearn_file)
            .filter(lambda src_file: fnmatch.fnmatch(src_file, '*datalearn.*.bin'))
            # RDD(record_dir), which is unique
            .distinct())

        self.run(root_dir, datalearn_file_rdd, origin_prefix)

    def run(self, root_dir, datalearn_file_rdd, origin_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(record_dir)
            datalearn_file_rdd
            # RDD(record_dir), in absolute path
            .map(lambda data_dir: os.path.join(root_dir, data_dir))
            # RDD(0/1), 1 for success
            .map(self.process_dir)
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(source_file):
        """Call prediction python code to combine features and labels"""
        source_dir = os.path.dirname(source_file)
        labels_dir = source_dir.replace('features-san-mateo', 'labels-san-mateo')
        label_file = os.path.join(labels_dir, 'junction_label.npy')
        CombineFeaturesAndLabels(source_file, label_file, 'junction')
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
            glog.info('Cannot find a feature-to-label mapping.')
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

        output_np_array.append(list_curr)

    output_np_array = np.array(output_np_array)

    np.save(feature_path + '.features+' + dict_name + '.npy', output_np_array)
    return output_np_array


if __name__ == '__main__':
    FeaturesAndLabelsCombine().run_prod()
