import argparse
import sys
import uuid

from absl import flags
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours
import google.protobuf.text_format as text_format

from modules.control.proto.control_conf_pb2 import ControlConf
from modules.control.proto.mrac_conf_pb2 import MracConf

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


flags.DEFINE_string(
    "tuner_param_config_filename",
    "fueling/learning/autotuner/config/mrac_tuner_param_config.pb.txt",
    "File path to tuner parameter config."
)
flags.DEFINE_string(
    "cost_computation_service_url",
    "localhost:50052",
    "URL to access the cost computation service"
)


def black_box_function(tuner_param_config_pb, algorithm_conf_pb):
    config_id = uuid.uuid1().hex
    CostComputationClient.set_channel(flags.FLAGS.cost_computation_service_url)
    weighted_score = CostComputationClient.compute_mrac_cost(
        tuner_param_config_pb.git_info.commit_id,
        {  # list of config_id : {path, config} pairs
            config_id:
            {tuner_param_config_pb.tuner_parameters.default_conf_filename: text_format.MessageToString(
                algorithm_conf_pb)},
        },
    )

    return weighted_score


class BayesianOptimizationTuner():
    """Basic functionality for NLP."""

    def __init__(self):
        logging.info(f"Init BayesianOptimization Tuner.")
        # Bounded region of parameter space
        self.pbounds = {}

        self.tuner_param_config_pb = TunerConfigs()
        self.algorithm_conf_pb = ControlConf()
        # Read and parse config from a pb file
        try:
            proto_utils.get_pb_from_text_file(
                flags.FLAGS.tuner_param_config_filename, self.tuner_param_config_pb,
            )
            logging.debug(f"Parsed autotune config files {self.tuner_param_config_pb}")

        except Exception as error:
            logging.error(f"Failed to parse autotune config: {error}")

        try:
            proto_utils.get_pb_from_text_file(
                self.tuner_param_config_pb.tuner_parameters.default_conf_filename, self.algorithm_conf_pb,
            )
            logging.debug(f"Parsed control config files {self.algorithm_conf_pb}")

        except Exception as error:
            logging.error(f"Failed to parse control config: {error}")

        for parameter in self.tuner_param_config_pb.tuner_parameters.parameter:
            self.pbounds.update({parameter.parameter_name: (parameter.min, parameter.max)})

        tuner_parameters = self.tuner_param_config_pb.tuner_parameters

        self.n_iter = tuner_parameters.n_iter

        self.utility = UtilityFunction(kind=tuner_parameters.utility.utility_name,
                                       kappa=tuner_parameters.utility.kappa,
                                       xi=tuner_parameters.utility.xi)

        self.optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=self.pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

    def set_bounds(self, bounds):
        self.pbounds = bounds

    def set_utility(self, kind, kappa, xi):
        self.utility = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def optimize(self, n_iter=5):
        self.n_iter = n_iter
        for i in range(n_iter):
            next_point = self.optimizer.suggest(self.utility)
            # TODO(QiL) extend to support tuning for repeated fields (repeated key in dict())
            self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf.MergeFrom(
                proto_utils.dict_to_pb(next_point, MracConf()))
            logging.debug(
                f"New Control Conf files {self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf}")
            target = black_box_function(self.tuner_param_config_pb, self.algorithm_conf_pb)
            self.optimizer.register(params=next_point, target=target)
            logging.debug(i, target, next_point)

    def get_result(self):
        logging.info(f"Result after: {self.n_iter} steps are  {self.optimizer.max}")
        return self.optimizer.max


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    tuner = BayesianOptimizationTuner()
    tuner.optimize()
    tuner.get_result()
    # TODO: dump back results to what path?
