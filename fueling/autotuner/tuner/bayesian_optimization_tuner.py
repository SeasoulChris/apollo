import argparse
import sys

from absl import flags
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours

import fueling.common.proto_utils as proto_utils
import fueling.common.logging as logging

from fueling.autotuner.client.cost_computation_client import CostComputationClient
import modules.data.fuel.fueling.autotuner.proto.tuner_param_config_pb2 as tuner_param_config_pb2


flags.DEFINE_string(
    "tuner_param_config_filename",
    "/apollo/modules/data/fuel/fueling/autotuner/config/tuner_param_config.pb.txt",
    "File path to tuner parameter config."
)


def black_box_function(tuner_param_config_pb, adaption_state_gain, adaption_matrix_p):
    weighted_score = CostComputationClient.compute_mrac_cost(

        tuner_param_config_pb.git_info.commit_id,
        # TODO: implement logics to generate updated parameters
        [  # list of {path, config} pairs
            {"apollo/modules/control/proto/control_conf.proto": "apollo/modules/control/conf/control.conf"},
        ],
    )

    return 2*adaption_state_gain + 3*adaption_matrix_p


class BayesianOptimizationTuner():
    """Basic functionality for NLP."""

    def __init__(self):
        logging.info(f"Init BayesianOptimization Tuner.")
        # Bounded region of parameter space
        # TODO: load bouns from configs

        self.utility = UtilityFunction(kind="ucb", kappa=3, xi=1)
        self.n_iter = 5
        self.tuner_param_config_pb = tuner_param_config_pb2.TunerConfigs()
        # Read and parse config from a pb file
        try:
            proto_utils.get_pb_from_text_file(
                flags.FLAGS.tuner_param_config_filename, self.tuner_param_config_pb,
            )
            logging.info(f"Parsed config files {self.tuner_param_config_pb}")

        #    self.pbounds = [
        #        proto_utils.pb_to_dict(parameter)["parameter"]
        #        for parameter in tuner_param_config_pb2.parameter
        #    ]

        except Exception as error:
            logging.error(f"Failed to parse config: {error}")

        self.pbounds = {'adaption_state_gain': (0, 2), 'adaption_matrix_p': (0, 3)}
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
            target = black_box_function(self.tuner_param_config_pb, **next_point)
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
