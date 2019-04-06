#!/usr/bin/env python

import glob
import os

import h5py
import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.s3_utils as s3_utils
import fueling.control.dynamic_model.offline_evaluator.model_evaluator as evaluator


class DynamicModelEvaluation(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'dynamic_model')

    def run_test(self):
        platform_path = '/apollo/modules/data/fuel/testdata/control/learning_based_model/'
        mlp_model_path = os.path.join(platform_path, 'dynamic_model_output/h5_model/mlp/*')
        lstm_model_path = os.path.join(platform_path, 'dynamic_model_output/h5_model/lstm/*')

        mlp_model_rdd = (
            # RDD(folder_path) for mlp models
            self.get_spark_context().parallelize(glob.glob(mlp_model_path))
            # PairRDD(model_name, folder_path)
            .keyBy(lambda _: 'mlp'))

        lstm_model_rdd = (
            # RDD(folder_path) for lstm models
            self.get_spark_context().parallelize(glob.glob(lstm_model_path))
            # PairRDD(model_name, folder_path)
            .keyBy(lambda _: 'lstm'))

        evaluation_dataset = [os.path.join(platform_path, 'hdf5_evaluation/evaluation.hdf5')]
        # RDD(file_path) for evaluation dataset
        evaluation_dataset_rdd = self.get_spark_context().parallelize(evaluation_dataset)

        self.model_evalution(mlp_model_rdd, evaluation_dataset_rdd, platform_path)
        self.model_evalution(lstm_model_rdd, evaluation_dataset_rdd, platform_path)

    def run_prod(self):
        platform_path = '/mnt/bos/modules/control/'
        bucket = 'apollo-platform'
        mlp_model_prefix = 'modules/control/dynamic_model_output/h5_model/mlp'
        lstm_model_prefix = 'modules/control/dynamic_model_output/h5_model/lstm'
        data_predix = 'modules/control/feature_extraction_hf5/hdf5_evaluation'

        mlp_model_rdd = (
            # RDD(folder_path) for mlp models
            s3_utils.list_dirs(bucket, mlp_model_prefix)
            # RDD(absolute_folder_path)
            .map(s3_utils.rw_path)
            # PairRDD(model_name, absolute_folder_path)
            .keyBy(lambda _: 'mlp'))

        lstm_model_rdd = (
            # RDD(folder_path) for lstm models
            s3_utils.list_dirs(bucket, lstm_model_prefix)
            # RDD(absolute_folder_path)
            .map(s3_utils.rw_path)
            # PairRDD(model_name, absolute_folder_path)
            .keyBy(lambda _: 'lstm'))

        evaluation_dataset_rdd = (
            # RDD(file_path) for evaluation dataset, which starts with data_predix
            s3_utils.list_files(bucket, data_predix)
            # RDD(file_path) for evaluation dataset, which ends with 'hdf5'
            .filter(lambda path: path.endswith('.hdf5'))
            # RDD(absolute_file_path)
            .map(s3_utils.rw_path))

        self.model_evalution(mlp_model_rdd, evaluation_dataset_rdd, platform_path)
        self.model_evalution(lstm_model_rdd, evaluation_dataset_rdd, platform_path)

    def model_evalution(self, model_rdd, evaluation_dataset_rdd, platform_path):
        results = (
            # PairRDD(dynamic_model_name, dynamic_model_path)
            model_rdd
            # PairRDD((dynamic_model_name, dynamic_model_path), evaluation_dataset_path)
            .cartesian(evaluation_dataset_rdd)
            # Action: call evaluation functions
            .foreach(lambda model_and_dataset: evaluator.evaluate(
                model_and_dataset[0], model_and_dataset[1], platform_path)))


if __name__ == '__main__':
    DynamicModelEvaluation().run_prod()
