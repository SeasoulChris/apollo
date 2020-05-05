#!/usr/bin/env python

import argparse
import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt

from modules.planning.proto import learning_data_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from fueling.planning.datasets.trajectory_perturbation_synthesizer import TrajectoryPerturbationSynthesizer


class TrajectoryPerturbationSynthesizerPipeline(BasePipeline):

    def __init__(self, src_dir, output_dir):
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.synthesizer = TrajectoryPerturbationSynthesizer()
        self.max_past_history_len = 10
        self.max_future_history_len = 10
        self.is_dumping_txt = False
        self.is_dumping_img = False

        # Make output_dir
        if os.path.isdir(self.output_dir):
            logging.info(self.output_dir +
                         " directory exists, delete it!")
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)
        logging.info("Making output directory: " + self.output_dir)

    def visualize_processed_frame(self, frame_file_path, past_trajectory, future_trajectory,
                                  perturbated_past_trajectory, perturbated_future_trajectory):
        origin_traj_for_plot = np.vstack(
            (past_trajectory, future_trajectory))
        traj_for_plot = np.vstack(
            (perturbated_past_trajectory, perturbated_future_trajectory))

        fig = plt.figure(0)
        xy_graph = fig.add_subplot(111)
        xy_graph.plot(
            origin_traj_for_plot[:, 0], origin_traj_for_plot[:, 1],  linestyle='--', marker='o', color='r')
        xy_graph.plot(traj_for_plot[:, 0], traj_for_plot[:, 1],
                      linestyle='--', marker='o', color='g')

        xy_graph.set_aspect('equal')
        plt.savefig(frame_file_path + ".jpg")

    def process_frame(self, frame_file_path):
        frame = proto_utils.get_pb_from_bin_file(
            frame_file_path, learning_data_pb2.LearningDataFrame())

        past_trajectory_length = len(frame.adc_trajectory_point) \
            if len(frame.adc_trajectory_point) < self.max_past_history_len \
            else self.max_past_history_len
        future_trajectory_length = len(frame.output.adc_future_trajectory_point) \
            if len(frame.output.adc_future_trajectory_point) < self.max_future_history_len \
            else self.max_future_history_len

        past_trajectory = np.zeros((past_trajectory_length, 3))
        future_trajectory = np.zeros((future_trajectory_length, 3))

        for i in range(past_trajectory_length):
            path_point = frame.adc_trajectory_point[len(frame.adc_trajectory_point) -
                                                    past_trajectory_length +
                                                    i].\
                trajectory_point.path_point
            past_trajectory[i, 0] = path_point.x
            past_trajectory[i, 1] = path_point.y
            past_trajectory[i, 2] = path_point.theta

        for i in range(future_trajectory_length):
            path_point = frame.output.adc_future_trajectory_point[i].trajectory_point.path_point
            future_trajectory[i, 0] = path_point.x
            future_trajectory[i, 1] = path_point.y
            future_trajectory[i, 2] = path_point.theta

        is_valid, perturbated_past_trajectory, perturbated_future_trajectory = \
            self.synthesizer.synthesize_perturbation(
                past_trajectory, future_trajectory)

        if not is_valid:
            return frame_file_path

        for i in range(past_trajectory_length):
            path_point = frame.adc_trajectory_point[len(frame.adc_trajectory_point) -
                                                    past_trajectory_length +
                                                    i].\
                trajectory_point.path_point
            path_point.x = perturbated_past_trajectory[i, 0]
            path_point.y = perturbated_past_trajectory[i, 1]
            path_point.theta = perturbated_past_trajectory[i, 2]

        for i in range(future_trajectory_length):
            path_point = frame.output.adc_future_trajectory_point[i].trajectory_point.path_point
            path_point.x = perturbated_future_trajectory[i, 0]
            path_point.y = perturbated_future_trajectory[i, 1]
            path_point.theta = perturbated_future_trajectory[i, 2]

        file_name = os.path.basename(frame_file_path)
        output_file_name = os.path.join(self.output_dir, file_name + '.synthesized.bin')
        proto_utils.write_pb_to_bin_file(frame, output_file_name)

        if self.is_dumping_txt:
            output_txt_name = output_file_name.replace('.bin', '') + '.txt'
            proto_utils.write_pb_to_text_file(frame, output_txt_name)

        if self.is_dumping_img:
            output_fig_name = output_file_name.replace('.bin', '') + '.png'
            self.visualize_processed_frame(output_fig_name,
                                           past_trajectory, 
                                           future_trajectory, 
                                           perturbated_past_trajectory, 
                                           perturbated_future_trajectory)

        return output_file_name

    def run_sequential(self):
        '''
        a process sequentially deal with .bin in LearningDataFrame
        '''
        start_time = time.time()
        logging.info('Processing directory: {}'.format(self.src_dir))
        all_file_paths = file_utils.list_files(self.src_dir)

        for file_path in all_file_paths:
            if 'future_status' not in file_path or 'bin' not in file_path:
                continue
            self.process_frame(file_path)
        logging.info('time spent is {}'.format(time.time() - start_time))

    def run(self):
        start_time = time.time()
        logging.info('Processing directory: {}'.format(self.src_dir))
        all_file_paths = file_utils.list_files(self.src_dir)

        file_paths_rdd = (
            # RDD(file_paths)
            self.to_rdd(all_file_paths)
            # RDD(filtered_file_paths), which filtered out non-bin or non-labeled file_path
            .filter(lambda file_path: 'future_status' in file_path and 'bin' in file_path)
            # RDD(processed_file_paths), which is the file_paths of processed files
            .map(self.process_frame)
            .collect())

        logging.info('time spent is {}'.format(time.time() - start_time))


if __name__ == "__main__":
    # TODO(Jinyun): use absl flag
    parser = argparse.ArgumentParser(description='source folder')
    parser.add_argument('src_dir', type=str,
                        help='data to be synthesized source folder')
    parser.add_argument('output_dir', type=str,
                        help='data to be synthesized source folder')
    args = parser.parse_args()

    synthesizer_pipeline = TrajectoryPerturbationSynthesizerPipeline(
        args.src_dir, args.output_dir)

    # synthesizer_pipeline.run_sequential()
    synthesizer_pipeline.main()
