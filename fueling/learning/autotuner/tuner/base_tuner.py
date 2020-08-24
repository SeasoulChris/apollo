from collections import namedtuple
from datetime import datetime
import glob
import json
import os
import shutil
import sys
import tarfile
import time
import uuid

from absl import flags
import google.protobuf.text_format as text_format
import numpy as np

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs
import fueling.common.email_utils as email_utils
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
flags.DEFINE_string(
    "running_role_postfix",
    "",
    "Optional postfix (e.g., mrac-control) of the running_role"
)


class BaseTuner():
    """Basic functionality for NLP."""

    def __init__(self, UserConfClassDict):
        tic_start = time.perf_counter()
        logging.info("Init Optimization Tuner.")

        self.tuner_param_config_pb, self.algorithm_conf_pb = self.read_configs(UserConfClassDict)

        # Bounded region or desired constant value of parameter space
        self.pbounds = {}
        self.pconstants = {}
        tuner_parameters = self.tuner_param_config_pb.tuner_parameters
        for parameter in tuner_parameters.parameter:
            if parameter.parameter_dir:
                parameter.parameter_name = parameter.parameter_dir + "." + parameter.parameter_name
            parameter = self.separate_repeated_param(parameter)
            if (parameter.min != parameter.max):
                self.pbounds.update({parameter.parameter_name: (parameter.min, parameter.max)})
            else:
                self.pconstants.update({parameter.parameter_name: parameter.constant})

        self.n_iter = tuner_parameters.n_iter

        self.init_points = tuner_parameters.init_points_1D ** (len(self.pbounds))
        self.init_params = self.initial_points(tuner_parameters.init_points_1D)

        self.opt_max = tuner_parameters.opt_max

        self.init_cost_client()

        self.tuner_storage_dir = (
            flags.FLAGS.tuner_storage_dir if os.path.isdir(flags.FLAGS.tuner_storage_dir)
            else '/fuel/testdata/autotuner'
        )

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.iteration_records = {}
        self.best_cost = 0.0
        self.best_params = {}
        self.optimize_time = 0.0
        self.time_efficiency = 0.0
        self.visual_storage_dir = self.get_saving_path()

        print(f"Training scenarios are {self.tuner_param_config_pb.scenarios.id}")

        self.init_optimizer_visualizer(self.tuner_param_config_pb.tuner_parameters)

        logging.info(f"Timer: initialize_tuner - {time.perf_counter() - tic_start: 0.04f} sec")

    def read_configs(self, UserConfClassDict):
        tuner_config_filename = flags.FLAGS.tuner_param_config_filename
        if not file_utils.file_exists(tuner_config_filename):
            raise Exception(f"No such config file found: {tuner_config_filename}")

        # Read and parse config from a pb file
        tuner_conf = TunerConfigs()
        try:
            proto_utils.get_pb_from_text_file(tuner_config_filename, tuner_conf)
            logging.debug(f"Parsed autotune config files {tuner_conf}")

        except Exception as error:
            logging.error(f"Failed to parse autotune config: {error}")
            sys.exit(1)

        user_module = tuner_conf.tuner_parameters.user_tuning_module  # user module Enum
        try:
            UserConfClass = UserConfClassDict[user_module]
            logging.debug(f"Assign user module proto {UserConfClass}")

        except Exception as error:
            logging.error(f"Failed to assign user module proto: {error}")
            sys.exit(1)

        user_conf = UserConfClass()  # Basic configuration corresponding to user module
        try:
            proto_utils.get_pb_from_text_file(
                tuner_conf.tuner_parameters.user_conf_filename, user_conf,
            )
            logging.debug(f"Parsed user config files {user_conf}")

        except Exception as error:
            logging.error(f"Failed to parse user config: {error}")
            sys.exit(1)

        return tuner_conf, user_conf

    def init_cost_client(self):
        config = self.tuner_param_config_pb

        self.cost_client = CostComputationClient(
            flags.FLAGS.cost_computation_service_url,
            config.git_info.commit_id,
            list(config.scenarios.id),
            config.dynamic_model,
            flags.FLAGS.running_role_postfix,
        )

    def init_optimizer_visualizer(self, tuner_parameters):
        "Optimizer and visualizer initialization algorithm"
        raise Exception("Not implemented!")

    def black_box_function(self, tuner_param_config_pb, algorithm_conf_pb):
        tic_start = time.perf_counter()
        """black box function for optimizers to implement the sim-tests and generate the costs"""
        config_id = uuid.uuid1().hex
        iteration_id, weighted_score = self.cost_client.compute_cost(
            {  # list of config_id : {path, config} pairs
                config_id:
                {tuner_param_config_pb.tuner_parameters.user_conf_filename:
                    text_format.MessageToString(algorithm_conf_pb)},
            }
        )
        logging.info(f"Received score for {iteration_id}")
        logging.info(f"Timer: sim_cost  - {time.perf_counter() - tic_start: 0.04f} sec")
        return iteration_id, weighted_score[config_id]

    def parse_param_to_proto(self, parameter_name):
        """Parse the parameters to generate all the elements in the form of protobf"""
        message_name = parameter_name.split('.')[0:-1]
        field_name = parameter_name.split('.')[-1]
        message = self.algorithm_conf_pb
        for name in message_name:
            message = getattr(message, name)
        # DESCRIPTOR attribute 'full_name' is formatted as
        # 'Package Name + Enclosing Type Name + Field name'
        # For example, 'apollo.control.LatControllerConf.matrix_q'
        config_name = message.DESCRIPTOR.fields_by_name[field_name].full_name.split('.')[-2]
        # DESCRIPTOR attribute 'label' is formatted as 'OPTIONAL = 1, REPEATED = 3, REQUIRED = 2'
        label = message.DESCRIPTOR.fields_by_name[field_name].label
        is_repeated = (label == 3)
        return message, config_name, field_name, is_repeated

    def separate_repeated_param(self, parameter):
        """Seqarate the repeated messages by adding the surffix '___digit' to their names"""
        _, _, _, is_repeated = self.parse_param_to_proto(parameter.parameter_name)
        if is_repeated:
            repeated_keys = ([key for key in self.pbounds if parameter.parameter_name in key]
                             + [key for key in self.pconstants if parameter.parameter_name in key])
            parameter.parameter_name += ('___' + str(len(repeated_keys)))
        return parameter

    def merge_repeated_param(self, point_origin):
        """Merge the separated message in a dict (config point) by eliminating the '___digit' """
        """and mergeing the values of repeated message into one list"""
        point = point_origin.copy()
        repeated_first_key = [key for key in point if '___0' in key]
        for first_key in repeated_first_key:
            base_key = first_key.replace('___0', '')
            repeated_keys = sorted([key for key in point if base_key in key])
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

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def optimize(self, n_iter=0, init_points=0):
        """Optimize core algorithm"""
        raise Exception("Not implemented!")

    def visualize(self, task_dir):
        """Visualize core algorithm"""
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

    def get_saving_path(self):
        return os.path.join(self.tuner_storage_dir, self.timestamp)

    def get_result(self):
        logging.info(f"Result after: {self.n_iter + self.init_points} steps are {self.best_cost} "
                     f"with params {self.best_params}")
        return (self.best_cost, self.best_params)

    def save_result(self):
        tic_start = time.perf_counter()
        self.tuner_results = {'best_target': self.best_cost,
                              'best_params': self.best_params,
                              'optimize_time': self.optimize_time,
                              'time_efficiency': self.time_efficiency}
        saving_path = self.get_saving_path()
        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)
        # Save multiple optimization data
        # (1) tuner_results.json file for internal developers
        with open(os.path.join(saving_path, "tuner_results.json"), 'w') as tuner_json:
            tuner_json.write(json.dumps(self.tuner_results))
        # (2) tuner_results.txt file for online service
        with open(os.path.join(saving_path, "tuner_results.txt"), 'w') as tuner_txt:
            tuner_txt.write('Control Parameters Auto-Tune Results: \n')
            for key in self.tuner_results.keys():
                if isinstance(self.tuner_results[key], dict):
                    tuner_txt.write(f'\n{key}: \n')
                    for subkey in self.tuner_results[key].keys():
                        tuner_txt.write(f'\t{subkey}: \t{self.tuner_results[key][subkey]} \n')
                else:
                    tuner_txt.write(f'\n{key}: \t{self.tuner_results[key]} \n')
        # (3) tuner_parameters.txt file for online service
        with open(os.path.join(saving_path, "tuner_parameters.txt"), 'w') as param_txt:
            param_txt.write(f'{self.tuner_param_config_pb.tuner_parameters}')
        # Save the final visual plots if the original figures are stored in single-iteration folder
        if saving_path != self.visual_storage_dir:
            final_visual_file = glob.glob(os.path.join(self.visual_storage_dir, '*.png'))
            for visual_file in final_visual_file:
                shutil.copyfile(
                    visual_file, os.path.join(
                        saving_path, os.path.basename(visual_file)))
        logging.info(f"Complete results saved at {saving_path}")
        self.summarize_task(saving_path)
        logging.info("Complete results summarized and released by email")
        logging.info(f"Timer: save_result  - {time.perf_counter() - tic_start: 0.04f} sec")

    def summarize_task(self, tuner_results_path, job_email='', error_msg=''):
        """Make summaries to specified task"""
        SummaryTuple = namedtuple(
            'Summary', [
                'Tuner_Config', 'Best_Target', 'Best_Params', 'Optimize_Time', 'Time_Efficiency'])
        title = 'Control Parameter Autotune Results'
        receivers = email_utils.DATA_TEAM + \
            email_utils.CONTROL_TEAM + email_utils.SIMULATION_TEAM
        receivers = []
        receivers.append(job_email)
        if tuner_results_path:
            email_content = []
            attachments = []
            # Initialize the attached files in reprot email
            target_file = glob.glob(os.path.join(
                tuner_results_path, '*tuner_results.txt'))
            target_conf = glob.glob(os.path.join(
                tuner_results_path, '*tuner_parameters.txt'))
            # Fill out the results summary in reprot email
            email_content.append(SummaryTuple(
                Tuner_Config=flags.FLAGS.tuner_param_config_filename,
                Best_Target=self.tuner_results['best_target'],
                Best_Params=self.tuner_results['best_params'],
                Optimize_Time=self.tuner_results['optimize_time'],
                Time_Efficiency=self.tuner_results['time_efficiency']))
            # Attach the result files in reprot email
            if target_file and target_conf:
                output_filename = os.path.join(
                    tuner_results_path,
                    f'control_autotune_{self.timestamp}.tar.gz')
                tar = tarfile.open(output_filename, 'w:gz')
                task_name = f'control_autotune_{self.timestamp}'
                file_name = os.path.basename(target_file[0])
                conf_name = os.path.basename(target_conf[0])
                tar.add(target_file[0], arcname=F'{task_name}_{file_name}')
                tar.add(target_conf[0], arcname=F'{task_name}_{conf_name}')
                tar.close()
                attachments.append(output_filename)
            file_utils.touch(os.path.join(tuner_results_path, 'COMPLETE'))
        else:
            logging.info('tuner_results_path in summarize_task: None')
            if error_msg:
                email_content = error_msg
            else:
                email_content = 'No autotune results: unknown reason.'
            attachments = []
        email_utils.send_email_info(
            title, email_content, receivers, attachments)

    def run(self):
        try:
            self.optimize()
            self.get_result()
        except Exception as error:
            logging.error(error)
        finally:
            self.save_result()
            self.cleanup()

    def cleanup(self):
        if self.cost_client:
            self.cost_client.close()
