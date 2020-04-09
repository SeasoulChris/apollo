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
from fueling.learning.autotuner.tuner.bayesian_optimization_visual_utils \
    import BayesianOptimizationVisualUtils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


flags.DEFINE_string(
    "tuner_param_config_filename",
    "",
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

class BaseTuner():
    """Basic functionality for NLP."""

    def __init__(self):
        logging.info(f"Init BayesianOptimization Tuner.")
        # Bounded region of parameter space
        self.pbounds = {}

        self.tuner_param_config_pb = TunerConfigs()

        # Read and parse config from a pb file
        try:
            proto_utils.get_pb_from_text_file(
                flags.FLAGS.tuner_param_config_filename, self.tuner_param_config_pb,
            )
            logging.debug(f"Parsed autotune config files {self.tuner_param_config_pb}")

        except Exception as error:
            logging.error(f"Failed to parse autotune config: {error}")

        tuner_parameters = self.tuner_param_config_pb.tuner_parameters
        #self.algorithm_conf_pb = tuner_parameters.default_conf_proto
        for parameter in tuner_parameters.parameter:
            parameter = self.separate_repeated_param(parameter)
            self.pbounds.update({parameter.parameter_name: (parameter.min, parameter.max)})

        self.n_iter = tuner_parameters.n_iter

        self.init_points = tuner_parameters.init_points_1D ** (len(self.pbounds))
        self.init_params = self.initial_points(tuner_parameters.init_points_1D)

        self.opt_max = tuner_parameters.opt_max

        self.utility = UtilityFunction(kind=tuner_parameters.utility.utility_name,
                                       kappa=tuner_parameters.utility.kappa,
                                       xi=tuner_parameters.utility.xi)

        #self.black_box_function = black_box_function(tuner_param_config_pb, algorithm_conf_pb)

        self.optimizer = BayesianOptimization(
            f=self.black_box_function,
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


    def black_box_function(self, tuner_param_config_pb, algorithm_conf_pb):
        config_id = uuid.uuid1().hex
        CostComputationClient.set_channel(flags.FLAGS.cost_computation_service_url)
        training_id, weighted_score = CostComputationClient.compute_mrac_cost(
            tuner_param_config_pb.git_info.commit_id,
            {  # list of config_id : {path, config} pairs
                config_id:
                {tuner_param_config_pb.tuner_parameters.default_conf_filename: text_format.MessageToString(
                    algorithm_conf_pb)},
            },
            list(tuner_param_config_pb.scenarios.id),
            tuner_param_config_pb.dynamic_model
        )
        logging.info(f"Received score for {training_id}")
        return training_id, weighted_score[config_id]

    def separate_repeated_param(self, parameter):
        """Seqarate the repeated messages by adding the surffix '___digit' to their names"""
        if parameter.is_repeated:
            repeated_keys = [key for key in self.pbounds if parameter.parameter_name in key]
            parameter.parameter_name += ('___' + str(len(repeated_keys)))
        return parameter

    def merge_repeated_param(self, point_origin):
        """Merge the separated message in a dict (config point) by eliminating the '___digit' """
        """and mergeing the values of repeated message into one list"""
        point = point_origin.copy()
        repeated_first_key = [key for key in point if '___0' in key]
        for first_key in repeated_first_key:
            base_key = first_key.replace('___0', '')
            repeated_keys = [key for key in point if base_key in key]
            repeated_keys.sort()
            point.update({base_key: [point[key] for key in repeated_keys]})
            for key in repeated_keys:
                del point[key]
        return point

    def initial_points(self, init_points_1D):
        init_grid_1D = {
            key: np.linspace(self.pbounds[key][0], self.pbounds[key][1], init_points_1D)
            for key in self.pbounds
        }
        input_grid_nD = np.array(np.meshgrid(*(init_grid_1D.values())))
        init_params = []
        for pts in range(self.init_points):
            init_params.append({list(self.pbounds)[idx]: input_grid_nD[idx].flatten()[pts]
                                for idx in range(len(self.pbounds))})
        return init_params

    def set_bounds(self, bounds):
        self.pbounds = bounds

    def set_utility(self, kind, kappa, xi):
        self.utility = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def optimize(self, n_iter=0, init_points=0):
            "OPtimize core algorithm"
            raise Exception("Not implemented!")

    def config_sanity_check(self, point_origin):
        point = point_origin.copy()
        param_name = list(point)[0]
        param_value = point[param_name]
        param_delta = (self.pbounds[param_name][1] - self.pbounds[param_name][0]) / 1000
        delta_sign = 1 if param_value <= sum(self.pbounds[param_name]) / 2 else -1

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
        if os.path.exists(final_visual_file):
            shutil.copyfile(final_visual_file, os.path.join(saving_path, 'gaussian_process.png'))
        logging.info(f"Detailed results saved at {saving_path} ")
