#!/usr/bin/env python
import glob
import math
import os

import colored_glog as glog
import numpy as np
import pyspark_utils.op as spark_op

from modules.prediction.proto import offline_features_pb2

from fueling.common.base_pipeline import BasePipeline

COLLISION_COST_EXP_COEFFICIENT = 1.0


class DataForTuningLabelsCombine(BasePipeline):
    """Records to DataForTuning proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'combine-data-for-tuning-and-labels')

    def run_test(self):
        """Run test."""
        datatuning_files = self.to_rdd(glob.glob(
            '/apollo/data/prediction/tuning/*/datatuning.*.bin'))
        self.run(datatuning_files)

    def run_prod(self):
        """Run prod."""
        origin_prefix = 'modules/prediction/tuning'

        # RDD(datatuning_file)
        datatuning_file_rdd = self.to_rdd(self.bos().list_files(origin_prefix)).filter(
            spark_op.filter_path(['*datatuning.*.bin']))

        self.run(datatuning_file_rdd)

    def run(self, datatuning_file_rdd):
        """Run the pipeline with given arguments."""
        # RDD(0/1), 1 for success
        result = datatuning_file_rdd.map(self.process_dir).count()
        glog.info('Processed {} tasks'.format(result))

    @staticmethod
    def process_dir(source_file):
        """Call prediction python code to combine features and labels"""
        source_dir = os.path.dirname(source_file)
        labels_dir = source_dir.replace('tuning', 'labels')
        label_file = os.path.join(labels_dir, 'future_status.npy')
        CombineDataForTuningAndLabels(source_file, label_file)
        return 0


def CombineDataForTuningAndLabels(feature_path, label_path):
    list_of_data_for_tuning = offline_features_pb2.ListDataForTuning()
    with open(feature_path, 'rb') as file_in:
        list_of_data_for_tuning.ParseFromString(file_in.read())
    dict_labels = np.load(label_path).item()

    glog.error(len(dict_labels))

    output_np_array = []
    for data_for_tuning in list_of_data_for_tuning.data_for_tuning:
        # values_for_tuning: list of doubles
        values_for_tuning = list(data_for_tuning.values_for_tuning)
        key = "{}@{:.3f}".format(data_for_tuning.id, data_for_tuning.timestamp)

        # Sanity checks to see if this data-point is valid or not.
        if key not in dict_labels:
            # glog.info('Cannot find a feature-to-label mapping.')
            continue

        labels = None
        list_curr = None
        future_status = dict_labels[key]
        start_timestamp = future_status[0][6]
        end_timestamp = future_status[-1][6]
        time_gap = end_timestamp - start_timestamp
        glog.error("time gap = " + str(time_gap))
        if time_gap < 2.1:
            continue

        adc_trajectory = data_for_tuning.adc_trajectory_point
        list_curr = values_for_tuning
        list_curr.append(ComputeLonAccCost(future_status))
        list_curr.append(ComputeCentripetalAccCost(future_status))
        list_curr.append(ComputeCollisionWithEgoVehicleCost(future_status, adc_trajectory))

        if list_curr is None:
            continue
        output_np_array.append(list_curr)

    output_np_array = np.array(output_np_array)

    np.save(feature_path + '.with_real_costs.npy', output_np_array)
    glog.error('file saved ' + feature_path + '.with_real_costs.npy')
    return output_np_array


def ComputeLonAccCost(future_status):
    costs = np.array([])
    time_stops = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    time_stop_index = 0
    future_status_index = 1
    start_timestamp = future_status[0][6]
    while future_status_index < len(future_status) and time_stop_index < len(time_stops):
        future_point = future_status[future_status_index]
        relative_time = future_point[6] - start_timestamp
        if relative_time > time_stops[time_stop_index]:
            prev_future_point = future_status[future_status_index - 1]
            delta_v = future_point[3] - prev_future_point[3]
            delta_t = future_point[6] - prev_future_point[6]
            acc = delta_v / (abs(delta_t) + 1e-6)
            costs = np.append(costs, acc * acc)
            time_stop_index += 1
        future_status_index += 1
    return np.mean(costs)


def ComputeCentripetalAccCost(future_status):
    cost_sqr_sum = 0.0
    cost_abs_sum = 0.0
    time_stops = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    time_stop_index = 0
    future_status_index = 1
    start_timestamp = future_status[0][6]
    while future_status_index < len(future_status) and \
          time_stop_index < len(time_stops):
        future_point = future_status[future_status_index]
        relative_time = future_point[6] - start_timestamp
        if relative_time > time_stops[time_stop_index]:
            previous_future_point = future_status[future_status_index - 1]
            delta_theta = future_point[2] - previous_future_point[2]
            delta_x = future_point[0] - previous_future_point[0]
            delta_y = future_point[1] - previous_future_point[1]
            delta_s = math.sqrt(delta_x * delta_x + delta_y * delta_y)
            centripetal_acc = delta_theta / (delta_s + 1e-6)
            cost_sqr_sum += centripetal_acc * centripetal_acc
            cost_abs_sum += abs(centripetal_acc)
            time_stop_index += 1
        future_status_index += 1
    return cost_sqr_sum / (cost_abs_sum + 1e-6)


def ComputeCollisionWithEgoVehicleCost(future_status, adc_trajectory):
    cost_sqr_sum = 0.0
    cost_abs_sum = 0.0
    adc_point_index = 0
    future_status_index = 0
    start_timestamp = future_status[0][6]
    while future_status_index < len(future_status) and \
          adc_point_index < len(adc_trajectory):
        future_point = future_status[future_status_index]
        relative_time = future_point[6] - start_timestamp
        if relative_time > adc_trajectory[adc_point_index].relative_time:
            adc_point = adc_trajectory[adc_point_index]
            delta_x = future_point[0] - adc_point.position.x
            delta_y = future_point[1] - adc_point.position.y
            distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
            cost = math.exp(-COLLISION_COST_EXP_COEFFICIENT * distance)
            cost_sqr_sum += cost * cost
            cost_abs_sum += abs(cost)
            adc_point_index += 1
        future_status_index += 1
    return cost_sqr_sum / (cost_abs_sum + 1e-6)


if __name__ == '__main__':
    DataForTuningLabelsCombine().main()
