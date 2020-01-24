from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours

from fueling.autotuner.client.cost_computation_client import CostComputationClient
import fueling.common.logging as logging


def black_box_function(x, y):
    weighted_score = CostComputationClient.compute_mrac_cost(
        # TODO: load commit id from parser
        # commit id
        "c693dd9e2e7910b041416021fcdb648cc4d8934d",
        [  # list of {path, config} pairs
            {"apollo/modules/control/proto/control_conf.proto": "apollo/modules/control/conf/control.conf"},
        ],
    )

    return 2*x + 3*y


class BayesianOptimizationTuner():
    """Basic functionality for NLP."""

    def __init__(self):
        logging.info(f"Init BayesianOptimization Tuner.")
        # Bounded region of parameter space
        # TODO: load bouns from configs
        self.pbounds = {'x': (2, 4), 'y': (-3, 3)}
        self.optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=self.pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        self.utility = UtilityFunction(kind="ucb", kappa=3, xi=1)
        self.n_iter = 5

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
            target = black_box_function(**next_point)
            self.optimizer.register(params=next_point, target=target)
            logging.debug(i, target, next_point)

    def get_result(self):
        logging.info(f"Result after: %s steps are %s", self.n_iter, self.optimizer.max)
        return self.optimizer.max


if __name__ == "__main__":
    tuner = BayesianOptimizationTuner()
    tuner.optimize()
    tuner.get_result()
    # TODO: dump back results to what path?
