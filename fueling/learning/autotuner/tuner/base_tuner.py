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

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs
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

    def __init__(self, tuner_conf, user_conf):
        logging.info(f"Init BayesianOptimization Tuner.")

        self.tuner_param_config_pb = tuner_conf
        self.algorithm_conf_pb = user_conf

        # Bounded region of parameter space
        self.pbounds = {}
        tuner_parameters = self.tuner_param_config_pb.tuner_parameters
        #self.algorithm_conf_pb = tuner_parameters.default_conf_proto
        for parameter in tuner_parameters.parameter:
            if parameter.parameter_dir:
                parameter.parameter_name = parameter.parameter_dir + "." + parameter.parameter_name
            parameter = self.separate_repeated_param(parameter)
            self.pbounds.update({parameter.parameter_name: (parameter.min, parameter.max)})

        self.n_iter = tuner_parameters.n_iter

        self.init_points = tuner_parameters.init_points_1D ** (len(self.pbounds))
        self.init_params = self.initial_points(tuner_parameters.init_points_1D)

        self.opt_max = tuner_parameters.opt_max

        self.init_cost_client()

        self.init_optimizer(tuner_parameters)

        self.tuner_storage_dir = (
            flags.FLAGS.tuner_storage_dir if os.path.isdir(flags.FLAGS.tuner_storage_dir)
            else 'testdata/autotuner'
        )

        print(f"Training scenarios are {self.tuner_param_config_pb.scenarios.id}")

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def init_cost_client(self):
        config = self.tuner_param_config_pb

        CostComputationClient.set_channel(flags.FLAGS.cost_computation_service_url)
        self.cost_client = CostComputationClient(
            config.git_info.commit_id,
            list(config.scenarios.id),
            config.dynamic_model,
        )

    def init_optimizer(self, tuner_parameters):
        self.utility = UtilityFunction(kind=tuner_parameters.utility.utility_name,
                                       kappa=tuner_parameters.utility.kappa,
                                       xi=tuner_parameters.utility.xi)

        self.optimizer = BayesianOptimization(
            f=self.black_box_function,
            pbounds=self.pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

    def black_box_function(self, tuner_param_config_pb, algorithm_conf_pb):
        config_id = uuid.uuid1().hex
        iteration_id, weighted_score = self.cost_client.compute_mrac_cost(
            {  # list of config_id : {path, config} pairs
                config_id:
                {tuner_param_config_pb.tuner_parameters.default_conf_filename: text_format.MessageToString(
                    algorithm_conf_pb)},
            }
        )
        logging.info(f"Received score for {iteration_id}")
        return iteration_id, weighted_score[config_id]

    def parse_param_to_proto(self, parameter_name):
        """Parse the parameters to generate all the elements in the form of protobf"""
        message_name = parameter_name.split('.')[0:-1]
        field_name = parameter_name.split('.')[-1]
        message = self.algorithm_conf_pb
        for name in message_name:
            message = getattr(message, name)
        # DESCRIPTOR attribute 'full_name' is formatted as 'Package Name + Enclosing Type Name + Field name'
        # For example, 'apollo.control.LatControllerConf.matrix_q'
        config_name = message.DESCRIPTOR.fields_by_name[field_name].full_name.split('.')[-2]
        # DESCRIPTOR attribute 'label' is formatted as 'OPTIONAL = 1, REPEATED = 3, REQUIRED = 2'
        label = message.DESCRIPTOR.fields_by_name[field_name].label
        is_repeated = True if label is 3 else False
        return message, config_name, field_name, is_repeated

    def separate_repeated_param(self, parameter):
        """Seqarate the repeated messages by adding the surffix '___digit' to their names"""
        _, _, _, is_repeated = self.parse_param_to_proto(parameter.parameter_name)
        if is_repeated:
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
        "Optimize core algorithm"
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

    def run(self):
        try:
            self.optimize()
            self.get_result()
            self.save_result()
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cost_client:
            self.cost_client.close()
