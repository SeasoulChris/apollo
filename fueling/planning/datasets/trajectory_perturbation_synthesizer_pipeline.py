#!/usr/bin/env python

import os
import shutil
import time

from absl import flags
import numpy as np

from modules.planning.proto import learning_data_pb2

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from fueling.planning.datasets.trajectory_perturbation_synthesizer \
    import TrajectoryPerturbationSynthesizer

flags.DEFINE_string('src_dir', None, 'data to be synthesized source folder')
flags.DEFINE_string('output_dir', None, 'output data folder')
flags.DEFINE_integer('max_past_history_len', 10,
                     'points num in the past to be synthesized')
flags.DEFINE_integer('max_future_history_len', 10,
                     'points num in the future to be synthesized')
flags.DEFINE_bool('is_dumping_txt', False, 'whether dump pb txt for debug')
flags.DEFINE_bool('is_dumping_img', False, 'whether dump imgs for debug')


class TrajectoryPerturbationSynthesizerPipeline(BasePipeline):
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
            path_point = frame.adc_trajectory_point[len(frame.adc_trajectory_point)
                                                    - past_trajectory_length
                                                    + i].\
                trajectory_point.path_point
            past_trajectory[i, 0] = path_point.x
            past_trajectory[i, 1] = path_point.y
            past_trajectory[i, 2] = path_point.theta

        for i in range(future_trajectory_length):
            path_point = frame.output.adc_future_trajectory_point[i].trajectory_point.path_point
            future_trajectory[i, 0] = path_point.x
            future_trajectory[i, 1] = path_point.y
            future_trajectory[i, 2] = path_point.theta

        perturbate_xy_range = self.FLAGS.get('perturbate_xy_range')
        ref_cost = self.FLAGS.get('ref_cost')
        elastic_band_smoothing_cost = self.FLAGS.get(
            'elastic_band_smoothing_cost')
        max_curvature = self.FLAGS.get('max_curvature')

        synthesizer = TrajectoryPerturbationSynthesizer(perturbate_xy_range,
                                                        ref_cost,
                                                        elastic_band_smoothing_cost,
                                                        max_curvature)

        is_valid = False
        perturbated_past_trajectory = None
        perturbated_future_trajectory = None
        perturbate_point_idx = None
        loop_counter = 0
        while not is_valid:
            if loop_counter >= 10:
                logging.error(
                    "fail to perturbated trajectory of " + frame_file_path)
                return frame_file_path
            is_valid, perturbated_past_trajectory, perturbated_future_trajectory,\
                perturbate_point_idx = \
                synthesizer.synthesize_perturbation(
                    past_trajectory, future_trajectory)
            loop_counter += 1

        for i in range(past_trajectory_length):
            path_point = frame.adc_trajectory_point[len(frame.adc_trajectory_point)
                                                    - past_trajectory_length
                                                    + i].\
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
        output_file_name = os.path.join(
            self.output_dir, file_name + '.synthesized.bin')
        proto_utils.write_pb_to_bin_file(frame, output_file_name)

        if self.is_dumping_txt:
            output_txt_name = output_file_name.replace('.bin', '') + '.txt'
            proto_utils.write_pb_to_text_file(frame, output_txt_name)

        if self.is_dumping_img:
            output_fig_name = output_file_name.replace('.bin', '') + '.png'
            synthesizer.visualize_for_debug(output_fig_name,
                                            past_trajectory,
                                            future_trajectory,
                                            perturbated_past_trajectory,
                                            perturbated_future_trajectory,
                                            perturbate_point_idx)

        return output_file_name

    def run(self):
        self.src_dir = self.FLAGS.get('src_dir')
        self.output_dir = self.FLAGS.get('output_dir')
        self.max_past_history_len = self.FLAGS.get('max_past_history_len')
        self.max_future_history_len = self.FLAGS.get('max_future_history_len')
        self.is_dumping_txt = self.FLAGS.get('is_dumping_txt')
        self.is_dumping_img = self.FLAGS.get('is_dumping_img')

        # Make output_dir
        if os.path.isdir(self.output_dir):
            logging.info(self.output_dir
                         + " directory exists, delete it!")
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)
        logging.info("Making output directory: " + self.output_dir)

        start_time = time.time()
        logging.info('Processing directory: {}'.format(self.src_dir))
        all_file_paths = file_utils.list_files(self.src_dir)

        (
            # RDD(file_paths)
            self.to_rdd(all_file_paths)
            # RDD(filtered_file_paths), which filtered out non-bin or non-labeled file_path
            .filter(lambda file_path: 'future_status' in file_path and 'bin' in file_path)
            # RDD(processed_file_paths), which is the file_paths of processed files
            .map(self.process_frame)
            .collect())

        logging.info('time spent is {}'.format(time.time() - start_time))


if __name__ == "__main__":
    TrajectoryPerturbationSynthesizerPipeline().main()
