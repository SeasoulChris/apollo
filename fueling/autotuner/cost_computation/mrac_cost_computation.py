#!/usr/bin/env python

from datetime import datetime
import glob
import json
import os

import modules.data.fuel.fueling.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2

from fueling.autotuner.cost_computation.base_cost_computation import BaseCostComputation
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


class MracCostComputation(BaseCostComputation):
    def __init__(self):
        BaseCostComputation.__init__(self)
        self.model_configs = None
        # TODO(Yu): add judgement to decide which submit-job tool is needed
        self.submit_job_cmd = "python ./tools/submit-job-to-local.py"

    def init(self):
        BaseCostComputation.init(self)

        # Read and parse config from a pb file
        try:
            request_pb = cost_service_pb2.Request()
            proto_utils.get_pb_from_text_file(
                f"{self.get_temp_dir()}/request.pb.txt", request_pb,
            )

            self.model_configs = [
                proto_utils.pb_to_dict(config_pb)["model_config"]
                for config_pb in request_pb.config
            ]

        except Exception as error:
            logging.error(f"Failed to parse config: {error}")
            return False

        return True

    def generate_config_rdd(self):
        return self.to_rdd(self.model_configs)

    def calculate_individual_score(self, input):
        (key, bag_path) = input
        logging.info(f"Calculating score for: {bag_path}")
        # submit the profiling job
        profiling_func = f"fueling/profiling/control/multi_job_control_profiling_metrics.py"
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
            return (key, [float('nan'), 0])
        # extract the profiling score of the individual scenario
        profiling_grading_dir = glob.glob(os.path.join(bag_path, '*/*/*/*grading.json'))
        logging.info(f"Score file storage path: {profiling_grading_dir}")
        with open(profiling_grading_dir[0], 'r') as grading_json:
            grading = json.load(grading_json)
        #TODO(Yu): implement weighting under the AutoTuner instead of directly reading
        profiling_score = grading['weighted_score']
        logging.info(f"Profiling score for individual scenario: "
                     f"score={profiling_score[0]}, sample={profiling_score[1]}")
        return (key, profiling_score)

    def calculate_weighted_score(self, config_2_score):
        if not len(config_2_score):
            return float('nan')
        # Calculate weighted score through multiple scenarios
        total_score = 0.0
        total_sample = 0
        for (config, scores) in config_2_score:
            total_score += scores[0] * scores[1]
            total_sample += scores[1]
        avg_score = total_score / total_sample if total_sample > 0 else avg = float('nan')
        return avg_score


if __name__ == "__main__":
    MracCostComputation().main()
