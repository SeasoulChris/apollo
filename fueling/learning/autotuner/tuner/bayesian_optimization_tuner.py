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

from modules.control.proto.control_conf_pb2 import ControlConf
from modules.control.proto.lat_controller_conf_pb2 import LatControllerConf
from modules.control.proto.mrac_conf_pb2 import MracConf

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs
from fueling.learning.autotuner.tuner.bayesian_optimization_visual_utils import BayesianOptimizationVisual
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
flags.DEFINE_string(
    "tuner_storage_dir",
    "/mnt/bos/autotuner",
    "Tuner storage root directory"
)

def black_box_function(tuner_param_config_pb, algorithm_conf_pb):
    config_id = uuid.uuid1().hex
    CostComputationClient.set_channel(flags.FLAGS.cost_computation_service_url)
    training_id, weighted_score = CostComputationClient.compute_mrac_cost(
        tuner_param_config_pb.git_info.commit_id,
        {  # list of config_id : {path, config} pairs
            config_id:
            {tuner_param_config_pb.tuner_parameters.default_conf_filename: text_format.MessageToString(
                algorithm_conf_pb)},
        },
        list(tuner_param_config_pb.scenarios.id)
    )
    logging.info(f"Received score for {training_id}")
    return training_id, weighted_score[config_id]


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

        tuner_parameters = self.tuner_param_config_pb.tuner_parameters

        try:
            proto_utils.get_pb_from_text_file(
                tuner_parameters.default_conf_filename, self.algorithm_conf_pb,
            )
            logging.debug(f"Parsed control config files {self.algorithm_conf_pb}")

        except Exception as error:
            logging.error(f"Failed to parse control config: {error}")

        for parameter in tuner_parameters.parameter:
            parameter = self.separate_repeated_param(parameter)
            self.pbounds.update({parameter.parameter_name: (parameter.min, parameter.max)})

        self.n_iter = tuner_parameters.n_iter

        self.opt_max = tuner_parameters.opt_max

        self.utility = UtilityFunction(kind=tuner_parameters.utility.utility_name,
                                       kappa=tuner_parameters.utility.kappa,
                                       xi=tuner_parameters.utility.xi)

        self.optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=self.pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        self.tuner_storage_dir = (
            flags.FLAGS.tuner_storage_dir if os.path.isdir(flags.FLAGS.tuner_storage_dir)
                                          else 'testdata/autotuner'
        )

        print(f"Training scenarios are {self.tuner_param_config_pb.scenarios.id}")

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def separate_repeated_param(self, parameter):
        """Seqarate the repeated messages by adding several surffix '__I' to their names"""
        i = 0
        while (i < len(self.pbounds)):
            if list(self.pbounds)[i] == parameter.parameter_name:
                parameter.parameter_name += '__I'
                i = 0
            else:
                i += 1
        return parameter

    def merge_repeated_param(self, point_origin):
        """Merge the separated message in a dict (config point) by eliminating the '__I' """
        """and mergeing the values of repeated message into one list"""
        point = point_origin.copy()
        repeated = []
        for key in point:
            if '__I' in key:
                repeated.append((key.count('__I'), key))
        repeated.sort()
        for number, key in repeated:
            base_key = key.replace('__I', '')
            if number == 1:
                point[base_key] = [point[base_key], point[key]]
            else:
                point[base_key].append(point[key])
            del point[key]
        return point

    def set_bounds(self, bounds):
        self.pbounds = bounds

    def set_utility(self, kind, kappa, xi):
        self.utility = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def optimize(self, n_iter=0):
        self.n_iter = n_iter if n_iter > 0 else self.n_iter
        self.iteration_records = {}
        visual = BayesianOptimizationVisual()
        for i in range(self.n_iter):
            next_point = self.config_sanity_check(self.optimizer.suggest(self.utility))

            for flag in self.tuner_param_config_pb.tuner_parameters.flag:
                self.algorithm_conf_pb.lat_controller_conf.MergeFrom(
                    proto_utils.dict_to_pb({flag.flag_name: flag.enable}, LatControllerConf()))
            self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf.ClearField(list(next_point)[0])
            self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf.MergeFrom(
                proto_utils.dict_to_pb(self.merge_repeated_param(next_point), MracConf()))
            logging.info(f"Enable MRAC control: "
                         f"{self.algorithm_conf_pb.lat_controller_conf.enable_steer_mrac_control}")
            logging.info(f"New MRAC Conf files: "
                         f"{self.algorithm_conf_pb.lat_controller_conf.steer_mrac_conf}")

            training_id, score = black_box_function(self.tuner_param_config_pb, self.algorithm_conf_pb)
            target = score if self.opt_max else -score
            self.optimizer.register(params=next_point, target=target)

            self.visual_storage_dir = os.path.join(self.tuner_storage_dir, training_id)
            if len(self.pbounds) == 1:
                param_name = list(self.pbounds)[0]
                visual.plot_gp(self.optimizer, self.utility, self.pbounds,
                               self.visual_storage_dir, param_name)

            self.iteration_records.update({f'iter-{i}': {'training_id': training_id, 'target': target,
                                                         'config_point': next_point}})

            logging.info(f"Optimizer iteration: {i}, target: {target}, config point: {next_point}")

    def config_sanity_check(self, point_origin):
        point = point_origin.copy()
        param_name = list(point)[0]
        param_value = point[param_name]
        param_delta = (self.pbounds[param_name][1] - self.pbounds[param_name][0]) / 1000
        delta_sign = (param_value <= (self.pbounds[param_name][1] + self.pbounds[param_name][0]) / 2)

        iter = 0
        while (iter < len(self.iteration_records)):
            if self.iteration_records[f'iter-{iter}']['config_point'] == point:
                param_value = point[param_name]
                point[param_name] = param_value + param_delta * delta_sign
                iter = 0
                logging.info(f"The config prameter {param_name} is adjusted from {param_value} "
                             f"to {point[param_name]} to fix the repeated config samples")
            else:
                iter += 1
        return point

    def get_result(self):
        logging.info(f"Result after: {self.n_iter} steps are  {self.optimizer.max}")
        return self.optimizer.max

    def save_result(self):
        tuner_param_config_dict = proto_utils.pb_to_dict(self.tuner_param_config_pb)
        self.tuner_results = {'target_max': self.optimizer.max['target'],
                              'config_max': self.optimizer.max['params'],
                              'tuner_parameters': tuner_param_config_dict['tuner_parameters'],
                              'iteration_records': self.iteration_records}

        saving_path = os.path.join(self.tuner_storage_dir, self.timestamp)
        os.makedirs(saving_path)
        with open(os.path.join(saving_path, "tuner_results.json"), 'w') as tuner_json:
            tuner_json.write(json.dumps(self.tuner_results))

        final_visual_file = os.path.join(self.visual_storage_dir, 'gaussian_process.png')
        if os.path.exist(final_visual_file):
            shutil.copyfile(final_visual_file, os.path.join(saving_path, 'gaussian_process.png'))
        logging.info(f"Detailed results saved at {saving_path} ")


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    tuner = BayesianOptimizationTuner()
    tuner.optimize()
    tuner.get_result()
    tuner.save_result()
