#!/usr/bin/env python

import glob
import json
import os
import sys
import time

from absl import flags

from apps.k8s.spark_submitter.client import SparkSubmitterClient
from fueling.learning.autotuner.cost_computation.base_cost_computation import BaseCostComputation
from fueling.learning.autotuner.proto.cost_computation_conf_pb2 import CostComputationConf
import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


flags.DEFINE_string(
    "cost_computation_conf_filename",
    "fueling/learning/autotuner/config/control_cost_computation_conf.pb.txt",
    "File path to cost computation config."
)


class ProfilingCostComputation(BaseCostComputation):
    def __init__(self):
        BaseCostComputation.__init__(self)

    def init(self):
        BaseCostComputation.init(self)

        try:
            self.request_pb2 = self.read_request()
        except Exception as error:
            logging.error(f"Failed to parse config: {error}")
            return False

        try:
            self.cost_conf_pb = self.read_cost_config()
        except Exception as error:
            logging.error(f"Failed to parse config: {error}")
            return False

        return True

    def read_request(self):
        """Read and parse request from a pb file"""
        request_pb2 = cost_service_pb2.ComputeRequest()
        proto_utils.get_pb_from_text_file(
            f"{self.get_absolute_iter_dir()}/compute_request.pb.txt", request_pb2,
        )
        return request_pb2

    def read_cost_config(self):
        """Read and parse config from cost computation pb file"""
        config_file = file_utils.fuel_path(self.FLAGS.get('cost_computation_conf_filename'))
        cost_conf_pb = CostComputationConf()
        proto_utils.get_pb_from_text_file(config_file, cost_conf_pb)
        return cost_conf_pb

    def get_config_map(self):
        return {
            id: proto_utils.pb_to_dict(config_pb)["model_config"]
            for (id, config_pb) in self.request_pb2.config.items()
        }

    def get_dynamic_model(self):
        return self.request_pb2.dynamic_model

    def run_profiling_locally(self, options):
        metrics_file = file_utils.fuel_path(
            self.cost_conf_pb.cost_params.profiling_filename
        )
        option_strings = [f"--{name}={value}" for (name, value) in options.items()]
        cmd = f"cd {file_utils.fuel_path('.')}; python {metrics_file} {' '.join(option_strings)}"
        logging.info(f"Executing '{cmd}'")

        exit_code = os.system(cmd)
        return os.WEXITSTATUS(exit_code) == 0

    def calculate_individual_score(self, bag_path):
        """calculate the individual-scenario profiling score combined with multiple metrics"""
        tic_start = time.perf_counter()
        logging.info(f"Calculating score for: {bag_path}")

        # submit the profiling job
        options = {}
        for flag in self.cost_conf_pb.cost_params.path_flag:
            options.update({flag.flag_name: bag_path})
        for flag in self.cost_conf_pb.cost_params.feature_flag:
            options.update({flag.flag_name: flag.feature})
        for flag in self.cost_conf_pb.cost_params.mode_flag:
            options.update({flag.flag_name: flag.enable})

        if not self.run_profiling_locally(options):
            logging.error(f"Fail to submit the profiling job.")
            self.pause_to_debug()
            return [float('nan'), 0]
        logging.info(f"Timer: run_profiling_locally - {time.perf_counter() - tic_start: 0.04f} sec")

        # extract the profiling score of the individual scenario
        profiling_grading_dir = glob.glob(os.path.join(
            bag_path, self.cost_conf_pb.cost_params.grading_filename
        ))
        logging.info(f"Score file storage path: {profiling_grading_dir}")

        if not profiling_grading_dir:
            logging.error(f"Fail to acquire the profiling grading file "
                          f"under the path: {bag_path}")
            self.pause_to_debug()
            return [float('nan'), 0]
        else:
            with open(profiling_grading_dir[0], 'r') as grading_json:
                grading = json.load(grading_json)
            # Parse the profiling results and compute the combined weighted-score
            # TODO (Yu/Vivian/Una): find an appropriate way to switch XXX_profiling_weighting
            profiling_score = self.control_profiling_weighting(grading)
            logging.info(f"Profiling score for individual scenario: "
                         f"score={profiling_score[0]}, sample={profiling_score[1]}")

            logging.info(f"Timer: total calculate_individual_score  - "
                         f"{time.perf_counter() - tic_start: 0.04f} sec")
            return profiling_score

    def calculate_weighted_score(self, config_and_scores):
        """calculate the multi-scenario profiling score with sample size as weighting factor"""
        if not len(config_and_scores):
            self.pause_to_debug()
            return float('nan')

        # Calculate weighted score through multiple scenarios
        total_score = 0.0
        total_sample = 0
        for (config, scores) in config_and_scores:
            total_score += scores[0] * scores[1]
            total_sample += scores[1]
        avg_score = total_score / total_sample if total_sample > 0 else float('nan')
        return avg_score

    def control_profiling_weighting(self, grading):
        """
        calculate the control profiling score based on the selected cost metrics and
        control-specified parameter formats
        """
        score = 0.0
        weighting = 0.0
        sample = grading['total_time_usage'][1]

        # Read and parse config from control cost computation pb file
        cost_metrics = self.cost_conf_pb.cost_metrics

        # Parse and compute the weighting metrics from control profiling results
        for metrics in cost_metrics.weighting_metrics:
            if 'peak' in metrics.metrics_name:
                # for peak metrics, the grading format: [[score, timestamp], sample]
                score += grading[metrics.metrics_name][0][0] * metrics.weighting_factor
            else:
                # for other metrics, the grading format: [score, sample]
                score += grading[metrics.metrics_name][0] * metrics.weighting_factor
            weighting += metrics.weighting_factor
        score /= weighting

        # Parse and compute the penalty metrics from control profiling results
        for metrics in cost_metrics.penalty_metrics:
            score += (grading[metrics.metrics_name][0] * grading[metrics.metrics_name][1] *
                      metrics.penalty_score)

        # Parse and compute the fail metrics from control profiling results
        for metrics in cost_metrics.fail_metrics:
            if grading[metrics.metrics_name][0] > 0:
                score = cost_metrics.fail_score

        return (score, sample)


if __name__ == "__main__":
    ProfilingCostComputation().main()
