import argparse
import os
import sys
import time

from absl import flags
import numpy as np
import optuna
from optuna.samplers import TPESampler

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
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils


class OptunaOptimizationTuner(BaseTuner):
    """Basic functionality of Optuna Tuner for Control Module."""

    def __init__(self):
        BaseTuner.__init__(self, ControlConf)

    def init_optimizer_visualizer(self, tuner_parameters):
        self.optimizer = optuna.create_study(sampler=TPESampler())
        self.visualizer = optuna.visualization

    def optimize(self, n_iter=0, init_points=0):
        tic_start_overall = time.perf_counter()
        self.n_iter = n_iter if n_iter > 0 else self.n_iter
        self.init_points = 0
        self.iter = -1

        self.optimizer.optimize(self.objective, n_trials=self.n_iter)
        self.best_cost = self.optimizer.best_value
        self.best_params = self.optimizer.best_params
        self.optimize_time = time.perf_counter() - tic_start_overall
        self.time_efficiency = self.optimize_time / (self.n_iter + self.init_points)

        self.visualize(self.get_saving_path())

    def visualize(self, task_dir):
        tic_start = time.perf_counter()

        self.visual_storage_dir = task_dir
        file_utils.makedirs(self.visual_storage_dir)
        figure1 = self.visualizer.plot_optimization_history(self.optimizer)
        figure2 = self.visualizer.plot_contour(self.optimizer, params=list(self.pbounds.keys())[:2])
        figure1.show()
        figure2.show()
        figure1.write_image(f"{self.visual_storage_dir}/optimization_history.png")
        figure2.write_image(f"{self.visual_storage_dir}/contour.png")

        logging.info(f"Timer: visualize  - {time.perf_counter() - tic_start: 0.04f} sec")

    def objective(self, trial):
        tic_start = time.perf_counter()

        self.iter += 1
        next_point = {}
        for key in self.pbounds:
            next_point.update(
                {key: trial.suggest_uniform(key, self.pbounds[key][0], self.pbounds[key][1])}
            )
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

        self.iteration_records.update({f'iter-{self.iter}': {'iteration_id': iteration_id,
                                                             'target': target,
                                                             'config_point': next_point}})

        logging.info(f"Timer: optimize_with_sim_cost  - {time.perf_counter() "
                     f"- tic_start: 0.04f} sec")

        return target


if __name__ == "__main__":
    tic_start = time.perf_counter()

    flags.FLAGS(sys.argv)
    tuner = OptunaOptimizationTuner()
    tuner.run()

    logging.info(f"Timer: overall optuna tuning  - {time.perf_counter() - tic_start: 0.04f} sec")
