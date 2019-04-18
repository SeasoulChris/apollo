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
            self.context().parallelize(glob.glob(mlp_model_path))
            # PairRDD(model_name, folder_path)
            .keyBy(lambda _: 'mlp'))

        lstm_model_rdd = (
            # RDD(folder_path) for lstm models
            self.context().parallelize(glob.glob(lstm_model_path))
            # PairRDD(model_name, folder_path)
            .keyBy(lambda _: 'lstm'))

        evaluation_dataset = [os.path.join(platform_path, 'hdf5_evaluation/evaluation.hdf5')]
        # RDD(file_path) for evaluation dataset
        evaluation_dataset_rdd = self.context().parallelize(evaluation_dataset)

        self.model_evalution(mlp_model_rdd, evaluation_dataset_rdd, platform_path)
        self.model_evalution(lstm_model_rdd, evaluation_dataset_rdd, platform_path)

    def run_prod(self):
        bucket = 'apollo-platform'
        platform_path = 'modules/control/learning_based_model/'
        mlp_model_prefix = os.path.join(platform_path, 'dynamic_model_output/h5_model/mlp')
        lstm_model_prefix = os.path.join(platform_path, 'dynamic_model_output/h5_model/lstm')
        data_predix = os.path.join(platform_path, 'hdf5_evaluation'

        # PairRDD('mlp', folder_path)
        mlp_model_rdd = s3_utils.list_dirs(bucket, mlp_model_prefix).keyBy(lambda _: 'mlp')
        # PairRDD('lstm', folder_path)
        lstm_model_rdd = s3_utils.list_dirs(bucket, lstm_model_prefix).keyBy(lambda _: 'lstm')
        # RDD(file_path) for evaluation dataset
        evaluation_dataset_rdd = s3_utils.list_files(bucket, data_predix, '.hdf5')

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
