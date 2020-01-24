#!/usr/bin/env python

import random
from datetime import datetime

import modules.data.fuel.fueling.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2

from fueling.autotuner.cost_computation.base_cost_computation import BaseCostComputation
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


class MracCostComputation(BaseCostComputation):
    def __init__(self):
        BaseCostComputation.__init__(self)
        self.model_configs = None

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
        # TODO: implement me
        (key, bag_path) = input
        logging.info(f"Calculating score for: {bag_path}")
        random.seed(datetime.now())
        return (key, [random.random(), random.random()])

    def calculate_weighted_score(self, config_2_score):
        # TODO: implement me
        if not len(config_2_score):
            return float('nan')

        total = sum([sum(scores) for (config, scores) in config_2_score])
        avg = total / len(config_2_score)
        return avg


if __name__ == "__main__":
    MracCostComputation().main()
