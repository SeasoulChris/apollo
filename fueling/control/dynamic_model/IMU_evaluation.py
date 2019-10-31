#!/usr/bin/env python
""" local version """
import glob
import os
import re

import h5py
import numpy as np
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model.conf.model_config import feature_config
import fueling.common.logging as logging
import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction
import fueling.control.dynamic_model.offline_evaluator.non_holistic_model_evaluator as evaluator

IS_BACKWARD = feature_config["is_backward"]


def extract_scenario_name(dataset_path):
    """ extract scenario name """
    logging.info("data_path:{}".format(dataset_path))
    result = re.findall(r"golden_test_backward/(.+?)/1.hdf5", dataset_path)
    logging.info("hdf5 files:{}".format(result))
    return result


class IMUEvaluation(BasePipeline):

    def run_test(self):
        """run test"""
        platform_path = "/apollo/modules/data/fuel/testdata/control/"
        if IS_BACKWARD:
            evaluation_set = "golden_test_backward"

        evaluation_dataset = os.path.join(platform_path, evaluation_set, "*/*.hdf5")
        logging.info("evaluation_dataset: {}".format(evaluation_dataset))
        logging.info("files in evaluation_dataset: {}".format(glob.glob(evaluation_dataset)))

        evaluation_dataset_rdd = (
            # RDD(file_path) for evaluation dataset
            self.to_rdd(glob.glob(evaluation_dataset))
            # PairRDD(driving_scenario, file_path) for evaluation dataset
            .keyBy(extract_scenario_name))
        logging.info(evaluation_dataset_rdd.collect())

        self.run(evaluation_dataset_rdd, platform_path)

    def run(self, evaluation_dataset_rdd, platform_path):
        """run"""
        results_rdd = spark_helper.cache_and_log(
            "results",
            # PairRDD(driving_scenario, file_path)
            evaluation_dataset_rdd
            # generate segment
            # PairRDD(driving_scenario, segment)
            .mapValues(feature_extraction.generate_segment)
            # processing segment
            # PairRDD(driving_scenario, (segment, segment_d, segment_dd))
            .mapValues(feature_extraction.IMU_feature_processing), 1)

        location = spark_helper.cache_and_log(
            "location",
            results_rdd
            # PairRDD(driving_scenario, (segment, segment_d, segment_dd))
            .map(lambda scenario_segments: evaluator.location(
                 scenario_segments, platform_path)), 1)


if __name__ == '__main__':
    IMUEvaluation().main()
