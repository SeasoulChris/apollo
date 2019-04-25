#!/usr/bin/env python
import fnmatch
import glob
import operator
import os

import colored_glog as glog
import numpy as np
import pyspark_utils.op as spark_op

from modules.prediction.proto import offline_features_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.bos_client as bos_client


TIME_RANGE = 3.0
DISTANCE_THRESHOLD = 1.5

class PerformanceEvaluator(BasePipeline):
    """Evaluate performace pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'evaluate-performance')

    def run_test(self):
        """Run test."""
        result_files = self.to_rdd(
            glob.glob('/apollo/data/mini_data_pipeline/results/*/prediction_result.*.bin'))
        metrics = self.run(result_files)
        saved_filename = 'metrics_{}.npy'.format(TIME_RANGE)
        save_path = os.path.join('/apollo/data/mini_data_pipeline/results', saved_filename)
        np.save(save_path, metrics)

    def run_prod(self):
        """Run prod."""
        origin_prefix = 'modules/prediction/results'
        # RDD(file) result files with the pattern prediction_result.*.bin
        result_file_rdd = self.to_rdd(self.bos().list_files(origin_prefix)).filter(
            spark_op.filter_path(['prediction_result.*.bin']))
        metrics = self.run(result_file_rdd)
        saved_filename = 'metrics_' + str(TIME_RANGE) + '.npy'
        np.save(os.path.join(bos_client.abs_path('modules/prediction/results'), saved_filename))

    def run(self, result_file_rdd):
        """Run the pipeline with given arguments."""
        # list [(unique_metric_key, metric_sum)]
        metrics = (
            # RDD(file), files with prediction results
            result_file_rdd
            # PairRDD(metric_key, metric_value)
            .flatMap(self.evaluate)
            # PairRDD(unique_metric_key, metric_sum)
            .reduceByKey(lambda v1, v2: v1 + v2)
            # list [(unique_metric_key, metric_sum)]
            .collect())
        return metrics

    @staticmethod
    def evaluate(result_file):
        """Call prediction python code to evaluate performance"""
        result_dir = os.path.dirname(result_file)
        future_status_dir = result_dir.replace('results', 'ground_truth')
        future_status_file = os.path.join(future_status_dir, 'future_status.npy')
        future_status_dict = np.load(future_status_file).item()

        portion_correct_predicted_sum = 0.0
        num_obstacle_sum = 0.0
        num_trajectory_sum = 0.0

        list_prediction_result = offline_features_pb2.ListPredictionResult()
        with open(result_file, 'rb') as f:
            list_prediction_result.ParseFromString(f.read())
        for prediction_result in list_prediction_result.prediction_result:
            portion_correct_predicted, num_obstacle, num_trajectory = \
                CorrectlyPredictePortion(prediction_result, future_status_dict, TIME_RANGE)
            portion_correct_predicted_sum += portion_correct_predicted
            num_obstacle_sum += num_obstacle
            num_trajectory_sum += num_trajectory

        return [('portion_correct_predicted_sum', portion_correct_predicted_sum),
                ('num_obstacle_sum', num_obstacle_sum),
                ('num_trajectory_sum', num_trajectory_sum)]


def IsCorrectlyPredicted(future_point, curr_time, prediction_result):
    future_relative_time = future_point[6] - curr_time
    for predicted_traj in prediction_result.trajectory:
        i = 0
        while i + 1 < len(predicted_traj.trajectory_point) and \
              predicted_traj.trajectory_point[i + 1].relative_time < future_relative_time:
            i += 1
        predicted_x = predicted_traj.trajectory_point[i].path_point.x
        predicted_y = predicted_traj.trajectory_point[i].path_point.y
        diff_x = abs(predicted_x - future_point[0])
        diff_y = abs(predicted_y - future_point[1])
        if diff_x < DISTANCE_THRESHOLD and diff_y < DISTANCE_THRESHOLD:
            return True
    return False


def CorrectlyPredictePortion(prediction_result, future_status_dict, time_range):
    dict_key = "{}@{:.3f}".format(prediction_result.id, prediction_result.timestamp)
    if dict_key not in future_status_dict:
        return 0.0, 0.0, 0.0
    obstacle_future_status = future_status_dict[dict_key]
    if not obstacle_future_status:
        return 0.0, 0.0, 0.0

    portion_correct_predicted = 0.0
    curr_timestamp = obstacle_future_status[0][6]

    total_future_point_count = 0.0
    correct_future_point_count = 0.0
    for future_point in obstacle_future_status:
        if future_point[6] - curr_timestamp > time_range:
            break
        if IsCorrectlyPredicted(future_point, curr_timestamp, prediction_result):
            correct_future_point_count += 1.0
        total_future_point_count += 1.0
    if total_future_point_count == 0:
        return 0.0, 0.0, 0.0
    portion_correct_predicted = correct_future_point_count / total_future_point_count

    return portion_correct_predicted, 1.0, len(prediction_result.trajectory)


if __name__ == '__main__':
    PerformanceEvaluator().main()
