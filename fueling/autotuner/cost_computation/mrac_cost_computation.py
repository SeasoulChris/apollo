#!/usr/bin/env python

from datetime import datetime
import glob
import json
import os
import sys

from absl import flags

from fueling.autotuner.cost_computation.base_cost_computation import BaseCostComputation
import fueling.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

# Flags
flags.DEFINE_enum(
    "profiling_running_mode", "TEST", ["TEST", "PROD"], "server running mode: TEST, PROD ."
)


class MracCostComputation(BaseCostComputation):
    def __init__(self):
        BaseCostComputation.__init__(self)

    def init(self):
        BaseCostComputation.init(self)

        if self.FLAGS.get('profiling_running_mode') == "PROD":
            self.submit_job_cmd = "python ./tools/submit-job-to-k8s.py --wait='True'"
        else:
            self.submit_job_cmd = "python ./tools/submit-job-to-local.py"

        try:
            self.request_pb2 = self.read_request()
        except Exception as error:
            logging.error(f"Failed to parse config: {error}")
            return False

        return True

    def read_request(self):
        """Read and parse request from a pb file"""
        request_pb2 = cost_service_pb2.Request()
        proto_utils.get_pb_from_text_file(
            f"{self.get_temp_dir()}/request.pb.txt", request_pb2,
        )
        return request_pb2

    def get_config_map(self):
        return {
            id: proto_utils.pb_to_dict(config_pb)["model_config"]
            for (id, config_pb) in self.request_pb2.config.items()
        }

    def calculate_individual_score(self, bag_path):
        logging.info(f"Calculating score for: {bag_path}")

        # submit the profiling job
        profiling_func = f"fueling/profiling/control/multi_job_control_profiling_metrics.py"
        if self.FLAGS.get('profiling_running_mode') == "PROD":
            profiling_flags = (f"--ctl_metrics_input_path_k8s={bag_path} "
                               f"--ctl_metrics_output_path_k8s={bag_path} "
                               f"--ctl_metrics_simulation_only_test='True' ")
        else:
            profiling_flags = (f"--ctl_metrics_input_path_local={bag_path} "
                               f"--ctl_metrics_output_path_local={bag_path} "
                               f"--ctl_metrics_simulation_only_test='True' ")
        profiling_cmd = (f"{self.submit_job_cmd} "
                         f"--main={profiling_func} --flags=\"{profiling_flags}\"")

        # verify exit status of the profiling job
        exit_code = os.system(profiling_cmd)
        if os.WEXITSTATUS(exit_code) != 0:
            logging.error(f"Fail to submit the control profiling job with "
                          f"error code {os.WEXITSTATUS(exit_code)}")
            return [float('nan'), 0]

        # extract the profiling score of the individual scenario
        profiling_grading_dir = glob.glob(os.path.join(bag_path, '*/*/*/*grading.json'))
        logging.info(f"Score file storage path: {profiling_grading_dir}")

        if not profiling_grading_dir:
            logging.error(f"Fail to acquire the control profiling grading .json file "
                          f"under the path: {bag_path}")
            return [float('nan'), 0]
        else:
            with open(profiling_grading_dir[0], 'r') as grading_json:
                grading = json.load(grading_json)
            # TODO(Yu): implement weighting under the AutoTuner instead of directly reading
            profiling_score = grading['weighted_score']
            logging.info(f"Profiling score for individual scenario: "
                         f"score={profiling_score[0]}, sample={profiling_score[1]}")
            return profiling_score

    def calculate_weighted_score(self, config_and_scores):
        if not len(config_and_scores):
            return float('nan')

        # Calculate weighted score through multiple scenarios
        total_score = 0.0
        total_sample = 0
        for (config, scores) in config_and_scores:
            total_score += scores[0] * scores[1]
            total_sample += scores[1]
        avg_score = total_score / total_sample if total_sample > 0 else float('nan')
        return avg_score


if __name__ == "__main__":
    MracCostComputation().main()
