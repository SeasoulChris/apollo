#!/usr/bin/env python
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
from modules.planning.proto import learning_data_pb2


class DataInspector:
    '''
    A data inspector to walk through learning_data file in a folder, and show some meta informations
    '''

    def __init__(self, data_dir):
        '''
        instances: every learning_data by frame
        total_num_instances: total learning_data num
        routing_existances: list of boolean whether routing by frame
        instance_timestamps: list of current time in sec given by localization by frame
        ego_future_timestamps: list of timestamps in sec of future points by frame
        ego_past_timestamps: list of timestamps in sec of future points by frame
        obstacles_future_timestamps: list of relative time in sec of every obstacle by frame.
            It's worth mentioned that absolute time recovered by prediction header time
        obstacles_past_timestamps: list of timestamps of every obstacle by frame
        '''
        self.instances = []
        self.total_num_instances = 0
        self.routing_existances = []
        self.instance_timestamps = []
        self.ego_future_timestamps = []
        self.ego_past_timestamps = []
        self.obstacles_future_timestamps = []
        self.obstacles_past_timestamps = []

        self._load_instances(data_dir)

        self._inspect_instances()

    def _load_instances(self, data_dir):
        logging.info('Processing directory: {}'.format(data_dir))
        all_file_paths = file_utils.list_files(data_dir)
        all_file_paths.sort(
            key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        for file_path in all_file_paths:
            if 'future_status' not in file_path or 'bin' not in file_path:
                continue
            logging.info("loading {} ...".format(file_path))
            learning_data_frames = learning_data_pb2.LearningData()
            with open(file_path, 'rb') as file_in:
                learning_data_frames.ParseFromString(file_in.read())
            for learning_data_frame in learning_data_frames.learning_data:
                self.instances.append(learning_data_frame)

        self.total_num_instances = len(self.instances)

        logging.info('Total number of data points = {}'.format(
            self.total_num_instances))

        for instance in self.instances:
            if len(instance.routing.local_routing_lane_id) == 0:
                self.routing_existances.append(False)
            else:
                self.routing_existances.append(True)

            self.instance_timestamps.append(instance.timestamp_sec)

            self.ego_future_timestamps.append(
                [adc_point.timestamp_sec for adc_point in instance.output.adc_future_trajectory_point])

            # Assuming ego past from learning_data is originally loaded in timestamp
            # ascending order, reverse it here to have newer points in front
            self.ego_past_timestamps.append(
                [adc_point.timestamp_sec for adc_point in reversed(instance.adc_trajectory_point)])

            self.obstacles_future_timestamps.append(
                [[[obs_point.relative_time + obstacle.obstacle_prediction.timestamp_sec
                   for obs_point in pred_trajectory.trajectory_point]
                  for pred_trajectory in obstacle.obstacle_prediction.trajectory]
                 for obstacle in instance.obstacle])
            # Assuming obs past from learning_data is originally loaded in timestamp
            # ascending order, reverse it here to have newer points in front
            self.obstacles_past_timestamps.append(
                [[obs_point.timestamp_sec for obs_point in reversed(obstacle.obstacle_trajectory_point)] for obstacle in instance.obstacle])

    def _inspect_instances(self):
        '''
        List of inspected items includes:
            1. Whether routing is given
            2. How delta_t is between labeled future trajectory and How long it is
            3. How delta_t is between past ego trajectory and How long it is
            4. How delta_t is average obstacle prediction trajectory and How long it is
            5. How delta_t is average obstacle past trajectory and How long it is
        Some inspections use for-loop rather than numpy array manipulation to inspect instances
            because the dimension of some lists loaded in _load_instances is not self consistent
        '''

        # Logging instance timestamps when there is no routing
        routing_nonexistances = np.array(
            [(routing_existance == False) + 0 for routing_existance in self.routing_existances])
        instance_timestamps = np.array(self.instance_timestamps)
        instances_without_routing = instance_timestamps[np.nonzero(
            routing_nonexistances * instance_timestamps)]
        logging.info('Total number of no routing instances are = {}'.format(
            instances_without_routing.size))
        if instances_without_routing.size != 0:
            logging.info('Timestamps in sec when there is no routing are {}'.format(
                instances_without_routing))

        ego_future_lengths = np.array(
            [len(ego_future) for ego_future in self.ego_future_timestamps])
        if not np.array_equal(ego_future_lengths,
                              np.ones(ego_future_lengths.shape, dtype=ego_future_lengths.dtype) * int(ego_future_lengths[0])):
            logging.info(
                'ego_future_length of different instance are NOT the same')
        else:
            logging.info(
                'ego_future_length of all instances is {} points'.format(ego_future_lengths[0]))

        # Note: Assuming first point in each future trajectory is the one delta_t ahead of current timestamp
        # ego_future_delta_t_info_by_frame consist of first point delta t to
        # current timestamp, average delta_t, max delta_t, min delta_t and total
        # time length
        ego_future_time_info = np.zeros([0, 5])
        for i, ego_future_timestamps in enumerate(self.ego_future_timestamps):
            ego_future_delta_t = []
            ego_future_delta_t.append(
                ego_future_timestamps[0] - self.instance_timestamps[i])
            for j in range(1, len(ego_future_timestamps)):
                ego_future_delta_t.append(
                    ego_future_timestamps[j] - ego_future_timestamps[j - 1])
            ego_future_delta_t = np.asarray(ego_future_delta_t)
            ego_future_time_info = np.vstack((ego_future_time_info,
                                              np.asarray([ego_future_delta_t[0],
                                                          np.mean(ego_future_delta_t),
                                                          np.max(ego_future_delta_t),
                                                          np.min(ego_future_delta_t),
                                                          np.sum(ego_future_delta_t)])))
        ego_future_time_info_average = np.mean(ego_future_time_info, axis=0)
        logging.info(
            'frame-averaged ego_future point delta_t between first future point to current time is {}'.format(
                ego_future_time_info_average[0]))
        logging.info(
            'frame-averaged ego_future average delta t is {}'.format(ego_future_time_info_average[1]))
        logging.info(
            'frame-averaged ego_future maximum delta_t is {}'.format(ego_future_time_info_average[2]))
        logging.info(
            'frame-averaged ego_future minimum delta_t is {}'.format(ego_future_time_info_average[3]))
        logging.info(
            'frame-averaged ego_future total time is {}'.format(ego_future_time_info_average[4]))

        ego_past_lengths = np.array([len(ego_past)
                                     for ego_past in self.ego_past_timestamps])
        if not np.array_equal(ego_past_lengths,
                              np.ones(ego_past_lengths.shape, dtype=ego_past_lengths.dtype) * int(ego_past_lengths[0])):
            logging.info(
                'ego_past_length of different instances are NOT the same')
        else:
            logging.info(
                'ego_past_length of all instances is {} points'.format(ego_past_lengths[0]))

        # Note: Assuming last point in each past trajectory is the one delta_t behind of current timestamp
        # ego_past_delta_t_info_by_frame consist of first point delta t to current
        # timestamp, average delta_t, max delta_t, min delta_t and total time
        # length
        ego_past_time_info = np.zeros([0, 5])
        for i, ego_past_timestamps in enumerate(self.ego_past_timestamps):
            ego_past_delta_t = []
            ego_past_delta_t.append(self.instance_timestamps[i] - ego_past_timestamps[0])
            for j in range(1, len(ego_past_timestamps)):
                ego_past_delta_t.append(
                    ego_past_timestamps[j - 1] - ego_past_timestamps[j])
            ego_past_delta_t = np.asarray(ego_past_delta_t)
            ego_past_time_info = np.vstack((ego_past_time_info,
                                            np.asarray([ego_past_delta_t[0],
                                                        np.mean(ego_past_delta_t),
                                                        np.max(ego_past_delta_t),
                                                        np.min(ego_past_delta_t),
                                                        np.sum(ego_past_delta_t)])))
        ego_past_time_info_average = np.mean(ego_past_time_info, axis=0)
        logging.info(
            'frame-averaged ego_past point delta_t between first past point to current time is {}'.format(
                ego_past_time_info_average[0]))
        logging.info(
            'frame-averaged ego_past average delta t is {}'.format(ego_past_time_info_average[1]))
        logging.info(
            'frame-averaged ego_past maximum delta_t is {}'.format(ego_past_time_info_average[2]))
        logging.info(
            'frame-averaged ego_past minimum delta_t is {}'.format(ego_past_time_info_average[3]))
        logging.info(
            'frame-averaged ego_past total time is {}'.format(ego_past_time_info_average[4]))

        obs_prediction_num = np.array([len(obs_prediction)
                                       for obs_prediction in self.obstacles_future_timestamps])
        logging.info('frame-averaged predicted obstacle num is {}'.format(np.mean(obs_prediction_num)))

        obs_future_time_info = np.zeros([0, 5])
        for i, obs_prediction in enumerate(self.obstacles_future_timestamps):
            current_timestamp = self.instance_timestamps[i]
            for obs in obs_prediction:
                for pred_trajectory in obs:
                    obs_future_delta_t = []
                    obs_future_delta_t.append(pred_trajectory[0] - current_timestamp)
                    for j in range(1, len(pred_trajectory)):
                        obs_future_delta_t.append(pred_trajectory[j] - pred_trajectory[j - 1])
                    obs_future_delta_t = np.asarray(obs_future_delta_t)
                    obs_future_time_info = np.vstack((obs_future_time_info, np.asarray([obs_future_delta_t[0],
                                                                                        np.mean(
                                                                                            obs_future_delta_t),
                                                                                        np.max(
                                                                                            obs_future_delta_t),
                                                                                        np.min(
                                                                                            obs_future_delta_t),
                                                                                        np.sum(obs_future_delta_t)])))
        obs_future_time_info_average = np.mean(obs_future_time_info, axis=0)
        logging.info(
            'frame-obs-multimodal-averaged obs_future point delta_t between first future point to current time is {}'.format(
                obs_future_time_info_average[0]))
        logging.info(
            'frame-obs-multimodal-averaged obs_future average delta t is {}'.format(obs_future_time_info_average[1]))
        logging.info(
            'frame-obs-multimodal-averaged obs_future maximum delta_t is {}'.format(obs_future_time_info_average[2]))
        logging.info(
            'frame-obs-multimodal-averaged obs_future minimum delta_t is {}'.format(obs_future_time_info_average[3]))
        logging.info(
            'frame-obs-multimodal-averaged obs_future total time is {}'.format(obs_future_time_info_average[4]))

        obs_past_time_info = np.zeros([0, 5])
        for i, obs_tracking in enumerate(self.obstacles_past_timestamps):
            current_timestamp = self.instance_timestamps[i]
            for obs_trajectory in obs_tracking:
                obs_past_delta_t = []
                obs_past_delta_t.append(current_timestamp - obs_trajectory[0])
                for j in range(1, len(obs_trajectory)):
                    obs_past_delta_t.append(obs_trajectory[j - 1] - obs_trajectory[j])
                obs_past_delta_t = np.asarray(obs_past_delta_t)
                obs_past_time_info = np.vstack((obs_past_time_info, np.asarray([obs_past_delta_t[0],
                                                                                np.mean(
                                                                                    obs_past_delta_t),
                                                                                np.max(
                                                                                    obs_past_delta_t),
                                                                                np.min(
                                                                                    obs_past_delta_t),
                                                                                np.sum(obs_past_delta_t)])))
        obs_past_time_info_average = np.mean(obs_past_time_info, axis=0)
        logging.info(
            'frame-obs-averaged obs_past point delta_t between first past point to current time is {}'.format(
                obs_past_time_info_average[0]))
        logging.info(
            'frame-obs-averaged obs_past average delta t is {}'.format(obs_past_time_info_average[1]))
        logging.info(
            'frame-obs-averaged obs_past maximum delta_t is {}'.format(obs_past_time_info_average[2]))
        logging.info(
            'frame-obs-averaged obs_past minimum delta_t is {}'.format(obs_past_time_info_average[3]))
        logging.info(
            'frame-obs-averaged obs_past total time is {}'.format(obs_past_time_info_average[4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data_inspector data folder')
    parser.add_argument('data_dir', type=str,
                        help='data_inspector data folder')
    args = parser.parse_args()

    data_inspector = DataInspector(args.data_dir)
