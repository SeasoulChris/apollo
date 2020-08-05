#!/usr/bin/env python
import glob
import os

import numpy as np

from modules.prediction.proto import offline_features_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.spark_op as spark_op


TIME_RANGES = [3.0, 1.0, 8.0]
REGIONS = ['sunnyvale', 'san_mateo']

DISTANCE_THRESHOLD = 1.5

FILTERED_EVALUATOR = None  # prediction_conf_pb2.ObstacleConf.SEMANTIC_LSTM_EVALUATOR
FILTERED_PREDICTOR = None  # prediction_conf_pb2.ObstacleConf.EXTRAPOLATION_PREDICTOR
FILTERED_PRIORITY = None  # feature_pb2.ObstaclePriority.CAUTION


class PerformanceEvaluator(BasePipeline):
    """Evaluate performance pipeline."""

    def run_test(self):
        """Run test."""
        result_files = self.to_rdd(
            glob.glob('/apollo/data/prediction/results/*/*/prediction_result.*.bin'))
        for time_range in TIME_RANGES:
            metrics = self.run_internal(result_files, time_range)
            saved_filename = 'metrics_{}.npy'.format(time_range)
            save_path = os.path.join('/apollo/data/prediction/results', saved_filename)
            np.save(save_path, metrics)

    def run(self):
        bos_client = self.our_storage()
        for time_range in TIME_RANGES:
            for region in REGIONS:
                """Run prod."""
                origin_prefix = os.path.join('modules/prediction/results', region)
                # RDD(file) result files with the pattern prediction_result.*.bin
                result_file_rdd = self.to_rdd(self.our_storage().list_files(origin_prefix)).filter(
                    spark_op.filter_path(['*prediction_result.*.bin']))
                metrics = self.run_internal(result_file_rdd, time_range)
                saved_filename = 'metrics_{}_{}.npy'.format(time_range, region)
                np.save(os.path.join(bos_client.abs_path('modules/prediction/results'),
                                     saved_filename), metrics)

    def run_internal(self, result_file_rdd, time_range):
        """Run the pipeline with given arguments."""
        # list [(unique_metric_key, metric_sum)]
        metrics = (
            # RDD(file), files with prediction results
            result_file_rdd
            # PairRDD(metric_key, metric_value)
            .flatMap(lambda file: self.evaluate(file, time_range))
            # PairRDD(unique_metric_key, metric_sum)
            .reduceByKey(lambda v1, v2: v1 + v2)
            # list [(unique_metric_key, metric_sum)]
            .collect())
        return metrics

    @staticmethod
    def evaluate(result_file, time_range):
        """Call prediction python code to evaluate performance"""
        result_dir = os.path.dirname(result_file)
        future_status_dir = result_dir.replace('results', 'labels')
        future_status_file = os.path.join(future_status_dir, 'future_status.npy')
        future_status_dict = np.load(future_status_file).item()

        portion_correct_predicted_sum = 0.0
        num_obstacle_sum = 0.0
        num_trajectory_sum = 0.0
        disp_sum = 0.0
        num_valid_disp = 0.0

        list_prediction_result = offline_features_pb2.ListPredictionResult()
        with open(result_file, 'rb') as f:
            list_prediction_result.ParseFromString(f.read())
        for prediction_result in list_prediction_result.prediction_result:
            if FILTERED_EVALUATOR is not None and \
               prediction_result.obstacle_conf.evaluator_type != FILTERED_EVALUATOR:
                continue
            if FILTERED_PREDICTOR is not None and \
               prediction_result.obstacle_conf.predictor_type != FILTERED_PREDICTOR:
                continue
            if FILTERED_PRIORITY is not None and \
               prediction_result.obstacle_conf.priority_type != FILTERED_PRIORITY:
                continue
            portion_correct_predicted, num_obstacle, num_trajectory = \
                CorrectlyPredictePortion(prediction_result, future_status_dict, time_range)
            portion_correct_predicted_sum += portion_correct_predicted
            num_obstacle_sum += num_obstacle
            num_trajectory_sum += num_trajectory

            disp = DisplacementError(prediction_result, future_status_dict, time_range)
            if disp is not None:
                print(disp)
                disp_sum += disp
                num_valid_disp += 1

        return [('portion_correct_predicted_sum', portion_correct_predicted_sum),
                ('num_obstacle_sum', num_obstacle_sum),
                ('num_trajectory_sum', num_trajectory_sum),
                ('disp_sum', disp_sum),
                ('num_valid_disp', num_valid_disp)]


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
        if diff_x * diff_x + diff_y * diff_y < DISTANCE_THRESHOLD * DISTANCE_THRESHOLD:
            return True
    return False


def CorrectlyPredictePortion(prediction_result, future_status_dict, time_range):
    dict_key = "{}@{:.3f}".format(prediction_result.id, prediction_result.timestamp)
    if dict_key not in future_status_dict:
        return 0.0, 0.0, 0.0
    obstacle_future_status = future_status_dict[dict_key]
    if not obstacle_future_status:
        return 0.0, 0.0, 0.0
    if len(prediction_result.trajectory) == 0:
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


def PointDisplacement(future_point, curr_time, prediction_result):
    disp_sum = 0.0
    num_valid_trajectory_point = 0.0
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
        disp_sum += np.sqrt(diff_x * diff_x + diff_y * diff_y)
        num_valid_trajectory_point += 1.0
    return (disp_sum, num_valid_trajectory_point)


def DisplacementError(prediction_result, future_status_dict, time_range):
    dict_key = "{}@{:.3f}".format(prediction_result.id, prediction_result.timestamp)
    if dict_key not in future_status_dict:
        return None
    obstacle_future_status = future_status_dict[dict_key]
    if not obstacle_future_status:
        return None
    if len(prediction_result.trajectory) == 0:
        return None

    curr_timestamp = obstacle_future_status[0][6]
    total_future_point_count = 0.0
    disp_error_sum = 0.0
    num_trajectory_point = 0.0
    for future_point in obstacle_future_status:
        if future_point[6] - curr_timestamp > time_range:
            break
        total_future_point_count += 1.0
        (disp_sum, num_valid_trajectory_point) = \
            PointDisplacement(future_point, curr_timestamp, prediction_result)
        disp_error_sum += disp_sum
        num_trajectory_point += num_valid_trajectory_point

    return disp_error_sum / (1e-6 + num_trajectory_point)


if __name__ == '__main__':
    PerformanceEvaluator().main()
