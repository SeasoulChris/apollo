#!/usr/bin/env python
""" local version """
import glob
import os
import re

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.spark_helper as spark_helper
import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction
import fueling.control.dynamic_model.offline_evaluator.non_holistic_model_evaluator as evaluator


def extract_scenario_name(dataset_path):
    """ extract scenario name """
    logging.info("data_path:{}".format(dataset_path))
    result = re.findall(r"golden_test_backward/(.+?)/1.hdf5", dataset_path)
    logging.info("hdf5 files:{}".format(result))
    return result


class IMUEvaluation(BasePipeline):

    def run(self):
        """run test"""
        is_backward = self.FLAGS.get('is_backward')
        platform_path = "/fuel/testdata/control/"
        if is_backward:
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

        self.run_internal(evaluation_dataset_rdd, platform_path)

    def run_internal(self, evaluation_dataset_rdd, platform_path):
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

        spark_helper.cache_and_log(
            "location",
            results_rdd
            # PairRDD(driving_scenario, (segment, segment_d, segment_dd))
            .map(lambda scenario_segments: evaluator.location(
                 scenario_segments, platform_path)), 1)


if __name__ == '__main__':
    IMUEvaluation().main()
