import argparse
import os
import sys
from datetime import datetime


from absl import flags
import numpy as np
import optuna
from optuna.samplers import TPESampler
import uuid
import google.protobuf.text_format as text_format


from modules.control.proto.control_conf_pb2 import ControlConf
from modules.control.proto.lat_controller_conf_pb2 import LatControllerConf
from modules.control.proto.lon_controller_conf_pb2 import LonControllerConf
from modules.control.proto.mpc_controller_conf_pb2 import MPCControllerConf
from modules.control.proto.gain_scheduler_conf_pb2 import GainScheduler
from modules.control.proto.leadlag_conf_pb2 import LeadlagConf
from modules.control.proto.mrac_conf_pb2 import MracConf
from modules.control.proto.pid_conf_pb2 import PidConf

from fueling.learning.autotuner.client.cost_computation_client import CostComputationClient
from fueling.learning.autotuner.proto.tuner_param_config_pb2 import TunerConfigs
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.file_utils as file_utils


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
flags.DEFINE_string(
    "visualization_dir",
    "/mnt/bos/autotuner/optuna",
    "Tuner storage root directory"
)
flags.DEFINE_string(
    "study_storage_url",
    "postgres:5432",
    "URL to access RDB"
)
flags.DEFINE_string(
    "study_name",
    "",
    "study name for optuna, this is necessary if running optuna in parallel. "
    "Otherwise, generate a random name."
)


class OptunaBaseTuner():
    """Basic functionality for Optuna Tuner."""

    def __init__(self):
        logging.info(f"Init OptunaBaseTuner Tuner.")

        tuner_conf = TunerConfigs()
        user_conf = ControlConf()  # Basic configuration corresponding to user module

        # Read and parse config from a pb file
        try:
            proto_utils.get_pb_from_text_file(
                file_utils.fuel_path(flags.FLAGS.tuner_param_config_filename), tuner_conf,
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

        self.tuner_param_config_pb = tuner_conf
        self.algorithm_conf_pb = user_conf

        # Bounded region of parameter space
        self.pbounds = {}

        self.init_cost_client()
        logging.info(f"Training scenarios are {self.tuner_param_config_pb.scenarios.id}")

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def objective(self, trial):
        config_id = uuid.uuid1().hex

        next_kp = trial.suggest_uniform('kp', 0, 10)
        next_ki = trial.suggest_uniform('ki', 0, 10)

        self.algorithm_conf_pb.lon_controller_conf.high_speed_pid_conf.kp = next_kp
        self.algorithm_conf_pb.lon_controller_conf.high_speed_pid_conf.ki = next_ki
        logging.info(f"Next data {next_kp}, {next_ki}")
        iteration_id, weighted_score = self.cost_client.compute_cost(
            {  # list of config_id : {path, config} pairs
                config_id:
                {self.tuner_param_config_pb.tuner_parameters.default_conf_filename:
                    text_format.MessageToString(self.algorithm_conf_pb)},
            }
        )
        logging.info(f"Received score for {iteration_id} as {weighted_score[config_id]}")
        return weighted_score[config_id]

    def init_cost_client(self):
        config = self.tuner_param_config_pb

        self.cost_client = CostComputationClient(
            flags.FLAGS.cost_computation_service_url,
            config.git_info.commit_id,
            list(config.scenarios.id),
            config.dynamic_model,
        )

    def close_cost_client(self):
        self.cost_client.close()


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    tuner = OptunaBaseTuner()

    n_iterations = tuner.tuner_param_config_pb.tuner_parameters.n_iter
    study_name = flags.FLAGS.study_name or f"autotuner-{tuner.timestamp}"
    logging.info(f"Running {study_name} for {n_iterations} trials...")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage="postgresql://{}:{}@{}/{}".format(
            os.environ["POSTGRES_USER"],
            os.environ["POSTGRES_PASSWORD"],
            flags.FLAGS.study_storage_url,
            os.environ["POSTGRES_DB"],
        ),
        sampler=TPESampler(),
        load_if_exists=True,
    )
    study.optimize(tuner.objective, n_trials=n_iterations)

    logging.info(f"Best value: {study.best_value}  (params: {study.best_params})")
    # optuna.visualization.plot_intermediate_values(study)

    output_dir = os.path.join(flags.FLAGS.visualization_dir, study_name, tuner.timestamp)
    file_utils.makedirs(output_dir)
    figure1 = optuna.visualization.plot_optimization_history(study)
    figure1.write_image(f"{output_dir}/optimization_history.png")
    figure2 = optuna.visualization.plot_contour(study, params=['kp', 'ki'])
    figure2.write_image(f"{output_dir}/contour.png")
    figure1.show()

    tuner.close_cost_client()
