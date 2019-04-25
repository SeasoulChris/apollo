#!/usr/bin/env python

import glob
import os
import re

import colored_glog as glog
import h5py
import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.s3_utils as s3_utils
import fueling.control.dynamic_model.offline_evaluator.model_evaluator as evaluator

VEHICLE_ID = 'Mkz7'

def extract_scenario_name(dataset_path):
    result = re.findall(r"hdf5_evaluation/.+/(.+?).hdf5", dataset_path)[0]
    return result


class DynamicModelEvaluation(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'dynamic_model')

    def run_test(self):
        platform_path = '/apollo/modules/data/fuel/testdata/control/learning_based_model/'
        mlp_model_path = os.path.join(platform_path, 'dynamic_model_output/h5_model/mlp/*')
        lstm_model_path = os.path.join(platform_path, 'dynamic_model_output/h5_model/lstm/*')

        # PairRDD(model_name, folder_path)
        mlp_model_rdd = self.to_rdd(glob.glob(mlp_model_path)).keyBy(lambda _: 'mlp')
        # PairRDD(model_name, folder_path)
        lstm_model_rdd = self.to_rdd(glob.glob(lstm_model_path)).keyBy(lambda _: 'lstm')

        evaluation_dataset = os.path.join(platform_path, 'hdf5_evaluation/%s/*.hdf5' % VEHICLE_ID)

        evaluation_dataset_rdd = (
            # RDD(file_path) for evaluation dataset
            self.to_rdd(glob.glob(evaluation_dataset))
            # PairRDD(driving_scenario, file_path) for evaluation dataset
            .keyBy(extract_scenario_name))

        self.model_evalution(mlp_model_rdd, evaluation_dataset_rdd, platform_path)
        self.model_evalution(lstm_model_rdd, evaluation_dataset_rdd, platform_path)

    def run_prod(self):
        bucket = 'apollo-platform'
        platform_path = 'modules/control/learning_based_model/'
        mlp_model_prefix = os.path.join(platform_path, 'dynamic_model_output/h5_model/mlp')
        lstm_model_prefix = os.path.join(platform_path, 'dynamic_model_output/h5_model/lstm')
        data_predix = os.path.join(platform_path, 'hdf5_evaluation/%s' % VEHICLE_ID)

        # PairRDD('mlp', folder_path)
        mlp_model_rdd = s3_utils.list_dirs(bucket, mlp_model_prefix).keyBy(lambda _: 'mlp')
        # PairRDD('lstm', folder_path)
        lstm_model_rdd = s3_utils.list_dirs(bucket, lstm_model_prefix).keyBy(lambda _: 'lstm')

        evaluation_dataset_rdd = (
            # RDD(file_path) for evaluation dataset
            s3_utils.list_files(bucket, data_predix, '.hdf5')
            # PairRDD(driving_scenario, file_path) for evaluation dataset
            .keyBy(extract_scenario_name))

        self.model_evalution(mlp_model_rdd, evaluation_dataset_rdd, platform_path)
        self.model_evalution(lstm_model_rdd, evaluation_dataset_rdd, platform_path)

    def model_evalution(self, model_rdd, evaluation_dataset_rdd, platform_path):
        results = (
            # PairRDD(dynamic_model_name, dynamic_model_path)
            model_rdd
            # PairRDD((dynamic_model_name, dynamic_model_path), 
            # (driving_scenario, evaluation_dataset_path))
            .cartesian(evaluation_dataset_rdd)
            # Action: call evaluation functions
            .foreach(lambda model_and_dataset: evaluator.evaluate(
                model_and_dataset[0], model_and_dataset[1], platform_path)))


if __name__ == '__main__':
    DynamicModelEvaluation().main()
