#!/usr/bin/env python

import glob
import os
import re

import colored_glog as glog
import h5py
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model.conf.model_config import feature_config
if feature_config["is_holistic"]:
    import fueling.control.dynamic_model.offline_evaluator.holistic_model_evaluator as evaluator
else:
    import fueling.control.dynamic_model.offline_evaluator.non_holistic_model_evaluator as evaluator

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

        evaluation_dataset = os.path.join(platform_path, 'hdf5_evaluation', VEHICLE_ID,
                                          'golden_test/*.hdf5')
        evaluation_dataset_rdd = (
            # RDD(file_path) for evaluation dataset
            self.to_rdd(glob.glob(evaluation_dataset))
            # PairRDD(driving_scenario, file_path) for evaluation dataset
            .keyBy(extract_scenario_name))

        self.model_evalution(mlp_model_rdd, evaluation_dataset_rdd, platform_path)
        self.model_evalution(lstm_model_rdd, evaluation_dataset_rdd, platform_path)

    def run_prod(self):
        platform_path = 'modules/control/learning_based_model/'
        mlp_model_prefix = os.path.join(platform_path, 'dynamic_model_output/h5_model/mlp')
        lstm_model_prefix = os.path.join(platform_path, 'dynamic_model_output/h5_model/lstm')
        data_predix = os.path.join(platform_path, 'hdf5_evaluation', VEHICLE_ID, 'golden_test')

        bos = self.bos()
        # PairRDD('mlp', folder_path)
        mlp_model_rdd = self.to_rdd(bos.list_dirs(mlp_model_prefix)).keyBy(lambda _: 'mlp')
        # PairRDD('lstm', folder_path)
        lstm_model_rdd = self.to_rdd(bos.list_dirs(lstm_model_prefix)).keyBy(lambda _: 'lstm')

        evaluation_dataset_rdd = (
            # RDD(file_path) for evaluation dataset
            self.to_rdd(bos.list_files(data_predix, '.hdf5'))
            # PairRDD(driving_scenario, file_path) for evaluation dataset
            .keyBy(extract_scenario_name))

        self.model_evalution(mlp_model_rdd, evaluation_dataset_rdd, platform_path)
        self.model_evalution(lstm_model_rdd, evaluation_dataset_rdd, platform_path)

    def model_evalution(self, model_rdd, evaluation_dataset_rdd, platform_path):
        results_rdd = (
            # PairRDD(dynamic_model_name, dynamic_model_path)
            model_rdd
            # PairRDD((dynamic_model_name, dynamic_model_path),
            # (driving_scenario, evaluation_dataset_path))
            .cartesian(evaluation_dataset_rdd)
            # Action: call evaluation functions
            # PairRDD(dynamic_model_path, trajectory_rmse)
            .flatMap(lambda model_and_dataset: evaluator.evaluate(
                model_and_dataset[0], model_and_dataset[1], platform_path))
            # PairRDD(dynamic_model_path, trajectory_rmse), which is valid
            .filter(lambda grading_result: grading_result is not None)
            # PairRDD(dynamic_model_path, iter[trajectory_rmse])
            .groupByKey()
            # PairRDD(dynamic_model_path, list[trajectory_rmse])
            .mapValues(list)
            # PairRDD(dynamic_model_path, average_trajectory_rmse)
            .mapValues(np.mean)
            .cache())

        def _print(result):
            dynamic_model_path, average_trajectory_rmse = result
            evaluation_result_path = os.path.join(dynamic_model_path, 'average_rmse.txt')
            with open(evaluation_result_path, 'w') as txt_file:
                txt_file.write('average rmse: {}'.format(average_trajectory_rmse))

        results_rdd.foreach(_print)


if __name__ == '__main__':
    DynamicModelEvaluation().main()
