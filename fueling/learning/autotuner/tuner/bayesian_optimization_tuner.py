import os
import sys
import time

from absl import flags
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

from modules.control.proto.control_conf_pb2 import ControlConf

from fueling.learning.autotuner.tuner.base_tuner import BaseTuner
from fueling.learning.autotuner.tuner.bayesian_optimization_visual_utils \
    import BayesianOptimizationVisualUtils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


class BayesianOptimizationTuner(BaseTuner):

    def __init__(self):
        BaseTuner.__init__(self, ControlConf)

    def init_optimizer_visualizer(self, tuner_parameters):
        self.utility = UtilityFunction(kind=tuner_parameters.utility.utility_name,
                                       kappa=tuner_parameters.utility.kappa,
                                       xi=tuner_parameters.utility.xi)

        self.optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=self.pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        self.visualier = BayesianOptimizationVisualUtils()

    def optimize(self, n_iter=0, init_points=0):
        tic_start_overall = time.perf_counter()
        self.n_iter = n_iter if n_iter > 0 else self.n_iter
        self.init_points = init_points if init_points > 0 else self.init_points
        for i in range(self.n_iter + self.init_points):
            tic_start = time.perf_counter()

            if i < self.init_points:
                next_point = self.config_sanity_check(self.init_params[i])
            else:
                next_point = self.config_sanity_check(self.optimizer.suggest(self.utility))
            for key in self.pconstants:
                next_point.update({key: self.pconstants[key]})

            next_point_pb = self.merge_repeated_param(next_point)

            for proto in next_point_pb:
                message, config_name, field_name, _ = self.parse_param_to_proto(proto)
                config = eval(config_name + '()')
                message.ClearField(field_name)
                message.MergeFrom(proto_utils.dict_to_pb(
                    {field_name: next_point_pb[proto]}, config))
                logging.info(f"\n  {proto}: {getattr(message, field_name)}")

            for flag in self.tuner_param_config_pb.tuner_parameters.flag:
                flag_full_name = \
                    flag.flag_dir + '.' + flag.flag_name if flag.flag_dir else flag.flag_name
                message, config_name, flag_name, _ = self.parse_param_to_proto(flag_full_name)
                config = eval(config_name + '()')
                message.ClearField(flag_name)
                message.MergeFrom(proto_utils.dict_to_pb({flag_name: flag.enable}, config))
                logging.info(f"\n  {flag_full_name}: {getattr(message, flag_name)}")

            iteration_id, score = self.black_box_function(
                self.tuner_param_config_pb, self.algorithm_conf_pb)
            target = score if self.opt_max else -score
            self.optimizer.register(params=next_point, target=target)

            self.iteration_records.update({f'iter-{i}': {'iteration_id': iteration_id,
                                                         'target': target,
                                                         'config_point': next_point}})

            logging.info(f"Timer: optimize_with_sim_cost  - "
                         f"{time.perf_counter() - tic_start: 0.04f} sec")
            logging.info(f"Optimizer iteration: {i}, target: {target}, config point: {next_point}")

            self.visualize(iteration_id)

        self.best_cost = self.optimizer.max['target']
        self.best_params = self.optimizer.max['params']
        self.optimize_time = time.perf_counter() - tic_start_overall
        self.time_efficiency = self.optimize_time / (self.n_iter + self.init_points)

    def visualize(self, task_dir):
        tic_start = time.perf_counter()

        self.visual_storage_dir = os.path.join(self.tuner_storage_dir, task_dir)
        self.visualier.plot_gp(self.optimizer, self.utility, self.pbounds, self.visual_storage_dir)

        logging.info(f"Timer: visualize  - {time.perf_counter() - tic_start: 0.04f} sec")


if __name__ == "__main__":
    tic_start = time.perf_counter()

    flags.FLAGS(sys.argv)
    tuner = BayesianOptimizationTuner()
    tuner.run()

    logging.info(f"Timer: overall bayesian tuning  - {time.perf_counter() - tic_start: 0.04f} sec")
