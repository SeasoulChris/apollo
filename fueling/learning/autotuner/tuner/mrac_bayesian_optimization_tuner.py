import argparse
from datetime import datetime
import json
import os
import shutil
import sys
import uuid

from absl import flags
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours
import google.protobuf.text_format as text_format
import numpy as np

from modules.control.proto.control_conf_pb2 import ControlConf
from modules.control.proto.lat_controller_conf_pb2 import LatControllerConf
from modules.control.proto.mrac_conf_pb2 import MracConf

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs
from fueling.learning.autotuner.tuner.base_tuner import BaseTuner
from fueling.learning.autotuner.tuner.bayesian_optimization_visual_utils \
    import BayesianOptimizationVisualUtils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils

class MRACBayesianOptimizationTuner(BaseTuner):
    """Basic functionality for NLP."""

    def __init__(self):
        BaseTuner.__init__(self)
        self.algorithm_conf_pb = ControlConf()
        try:
            proto_utils.get_pb_from_text_file(
                self.tuner_param_config_pb.tuner_parameters.default_conf_filename, self.algorithm_conf_pb,
            )
            logging.debug(f"Parsed control config files {self.algorithm_conf_pb}")

        except Exception as error:
            logging.error(f"Failed to parse control config: {error}")

    def optimize(self, n_iter=0, init_points=0):
        self.n_iter = n_iter if n_iter > 0 else self.n_iter
        self.init_points = init_points if init_points > 0 else self.init_points
        self.iteration_records = {}
        visual = BayesianOptimizationVisualUtils()
        for i in range(self.n_iter + self.init_points):
            if i < self.init_points:
                next_point = self.config_sanity_check(self.init_params[i])
            else:
                next_point = self.config_sanity_check(self.optimizer.suggest(self.utility))

            for flag in self.tuner_param_config_pb.tuner_parameters.flag:
                self.algorithm_conf_pb.lat_controller_conf.MergeFrom(
                    proto_utils.dict_to_pb({flag.flag_name: flag.enable}, LatControllerConf()))
            next_point_pb = self.merge_repeated_param(next_point)
            for field in next_point_pb:
                self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf.ClearField(field)
            self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf.MergeFrom(
                proto_utils.dict_to_pb(next_point_pb, MracConf()))
            logging.info(f"Enable MRAC control: "
                         f"{self.algorithm_conf_pb.lat_controller_conf.enable_steer_mrac_control}")
            logging.info(f"New MRAC Conf files: \n"
                         f"{self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf}")

            training_id, score = self.black_box_function(self.tuner_param_config_pb, self.algorithm_conf_pb)
            target = score if self.opt_max else -score
            self.optimizer.register(params=next_point, target=target)

            self.visual_storage_dir = os.path.join(self.tuner_storage_dir, training_id)
            visual.plot_gp(self.optimizer, self.utility, self.pbounds, self.visual_storage_dir)

            self.iteration_records.update({f'iter-{i}': {'training_id': training_id, 'target': target,
                                                         'config_point': next_point}})

            logging.info(f"Optimizer iteration: {i}, target: {target}, config point: {next_point}")

if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    tuner = MRACBayesianOptimizationTuner()
    tuner.optimize()
    tuner.get_result()
    tuner.save_result()
