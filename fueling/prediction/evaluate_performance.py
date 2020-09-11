#!/usr/bin/env python
import os

import numpy as np

from modules.prediction.proto import offline_features_pb2
from modules.perception.proto import perception_obstacle_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.spark_op as spark_op


TIME_RANGES = [3.0, 1.0]

FILTERED_OBSTACLE_TYPE = perception_obstacle_pb2.PerceptionObstacle.PEDESTRIAN
FILTERED_EVALUATOR = None  # prediction_conf_pb2.ObstacleConf.SEMANTIC_LSTM_EVALUATOR
FILTERED_PREDICTOR = None  # prediction_conf_pb2.ObstacleConf.EXTRAPOLATION_PREDICTOR
FILTERED_PRIORITY = None  # feature_pb2.ObstaclePriority.CAUTION


class PerformanceEvaluator(BasePipeline):
    """Evaluate performance pipeline."""
    def run(self):
        input_path = 'modules/prediction/kinglong_benchmark'
        self.result_prefix = os.path.join(input_path, 'results')
        self.label_prefix = os.path.join(input_path, 'results')
        self.object_storage = self.partner_storage() or self.our_storage()
        for time_range in TIME_RANGES:
            # RDD(file) result files with the pattern prediction_result.*.bin
            result_file_rdd = (
                self.to_rdd(self.our_storage().list_files(self.result_prefix))
                .filter(spark_op.filter_path(['*prediction_result.*.bin'])))
            metrics = self.run_internal(result_file_rdd, time_range)
            saved_filename = 'metrics_{}.npy'.format(time_range)
            np.save(os.path.join(self.object_storage.abs_path(self.result_prefix),
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

    def evaluate(self, result_file, time_range):
        """Call prediction python code to evaluate performance"""
        result_dir = os.path.dirname(result_file)
        future_status_dir = result_dir.replace('results', 'labels')
        future_status_filenames = os.listdir(future_status_dir)

        future_status_dict = dict()
        for future_status_filename in future_status_filenames:
            future_status_filepath = os.path.join(future_status_dir, future_status_filename)
            if future_status_filepath.endswith('future_status.npy'):
                dict_curr = np.load(future_status_filepath, allow_pickle=True).item()
                future_status_dict.update(dict_curr)

        total_ade_sum = 0.0
        total_num_trajectory = 0.0

        list_prediction_result = offline_features_pb2.ListPredictionResult()
        with open(result_file, 'rb') as f:
            list_prediction_result.ParseFromString(f.read())
        for prediction_result in list_prediction_result.prediction_result:
            if FILTERED_OBSTACLE_TYPE is not None and \
               prediction_result.obstacle_conf.obstacle_type != FILTERED_OBSTACLE_TYPE:
                continue
            if FILTERED_EVALUATOR is not None and \
               prediction_result.obstacle_conf.evaluator_type != FILTERED_EVALUATOR:
                continue
            if FILTERED_PREDICTOR is not None and \
               prediction_result.obstacle_conf.predictor_type != FILTERED_PREDICTOR:
                continue
            if FILTERED_PRIORITY is not None and \
               prediction_result.obstacle_conf.priority_type != FILTERED_PRIORITY:
                continue
            ade_sum, num_traj = self.DisplacementError(prediction_result,
                                                       future_status_dict, time_range)
            total_ade_sum += ade_sum
            total_num_trajectory += num_traj

        logging.info('(ade, num_traj) = ({}, {})'.format(total_ade_sum, total_num_trajectory))
        return [('total_ade_sum', total_ade_sum),
                ('total_num_trajectory', total_num_trajectory)]

    def DisplacementError(self, prediction_result, future_status_dict, time_range):
        dict_key = "{}@{:.3f}".format(prediction_result.id, prediction_result.timestamp)
        if dict_key not in future_status_dict:
            return 0.0, 0.0
        obstacle_future_status = future_status_dict[dict_key]
        if not obstacle_future_status:
            return 0.0, 0.0
        if len(prediction_result.trajectory) == 0:
            return 0.0, 0.0

        num_point = int(round(time_range / 0.1))

        if len(obstacle_future_status) < num_point + 1:
            return 0.0, 0.0

        ade_sum = 0.0
        num_trajectory = 0.0
        for predicted_traj in prediction_result.trajectory:
            if len(predicted_traj.trajectory_point) < num_point + 1:
                continue
            ade = 0.0
            for i in range(1, num_point + 1):
                pred_x = predicted_traj.trajectory_point[i].path_point.x
                pred_y = predicted_traj.trajectory_point[i].path_point.y
                true_x = obstacle_future_status[i][0]
                true_y = obstacle_future_status[i][1]
                diff_x = abs(pred_x - true_x)
                diff_y = abs(pred_y - true_y)
                ade += np.sqrt(diff_x * diff_x + diff_y * diff_y)
            ade /= num_point
            ade_sum += ade
            num_trajectory += 1

        return ade_sum, num_trajectory


if __name__ == '__main__':
    PerformanceEvaluator().main()
