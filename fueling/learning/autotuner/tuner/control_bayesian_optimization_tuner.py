import argparse
import os
import sys

from absl import flags
import numpy as np

from modules.control.proto.control_conf_pb2 import ControlConf
from modules.control.proto.lat_controller_conf_pb2 import LatControllerConf
from modules.control.proto.lon_controller_conf_pb2 import LonControllerConf
from modules.control.proto.mpc_controller_conf_pb2 import MPCControllerConf
from modules.control.proto.gain_scheduler_conf_pb2 import GainScheduler
from modules.control.proto.leadlag_conf_pb2 import LeadlagConf
from modules.control.proto.mrac_conf_pb2 import MracConf
from modules.control.proto.pid_conf_pb2 import PidConf

from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs
from fueling.learning.autotuner.tuner.base_tuner import BaseTuner
from fueling.learning.autotuner.tuner.bayesian_optimization_visual_utils \
    import BayesianOptimizationVisualUtils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


class ControlBayesianOptimizationTuner(BaseTuner):
    """Basic functionality for NLP."""

    def __init__(self):
        tuner_conf = TunerConfigs()
        user_conf = ControlConf()  # Basic configuration corresponding to user module

        # Read and parse config from a pb file
        try:
            proto_utils.get_pb_from_text_file(
                flags.FLAGS.tuner_param_config_filename, tuner_conf,
            )
            logging.debug(f"Parsed autotune config files {tuner_conf}")

        except Exception as error:
            logging.error(f"Failed to parse autotune config: {error}")
            sys.exit(1)

        try:
            proto_utils.get_pb_from_text_file(
                tuner_conf.tuner_parameters.default_conf_filename, user_conf,
            )
            logging.debug(f"Parsed user config files {user_conf}")

        except Exception as error:
            logging.error(f"Failed to parse user config: {error}")
            sys.exit(1)

        BaseTuner.__init__(self, tuner_conf, user_conf)

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

            next_point_pb = self.merge_repeated_param(next_point)

            for proto in next_point_pb:
                message, config_name, field_name, _ = self.parse_param_to_proto(proto)
                config = eval(config_name + '()')
                message.ClearField(field_name)
                message.MergeFrom(proto_utils.dict_to_pb({field_name: next_point_pb[proto]}, config))
                logging.info(f"\n  {proto}: {getattr(message, field_name)}")

            for flag in self.tuner_param_config_pb.tuner_parameters.flag:
                flag_full_name = flag.flag_dir + '.' + flag.flag_name if flag.flag_dir else flag.flag_name
                message, config_name, flag_name, _ = self.parse_param_to_proto(flag_full_name)
                config = eval(config_name + '()')
                message.ClearField(flag_name)
                message.MergeFrom(proto_utils.dict_to_pb({flag_name: flag.enable}, config))
                logging.info(f"\n  {flag_full_name}: {getattr(message, flag_name)}")

            iteration_id, score = self.black_box_function(
                self.tuner_param_config_pb, self.algorithm_conf_pb)
            target = score if self.opt_max else -score
            self.optimizer.register(params=next_point, target=target)

            self.visual_storage_dir = os.path.join(self.tuner_storage_dir, iteration_id)
            visual.plot_gp(self.optimizer, self.utility, self.pbounds, self.visual_storage_dir)

            self.iteration_records.update({f'iter-{i}': {'iteration_id': iteration_id, 'target': target,
                                                         'config_point': next_point}})

            logging.info(f"Optimizer iteration: {i}, target: {target}, config point: {next_point}")


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    tuner = ControlBayesianOptimizationTuner()
    tuner.run()
