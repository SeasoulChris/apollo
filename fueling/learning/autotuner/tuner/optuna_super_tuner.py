import os
import sys
import time
import math


from absl import flags
import optuna
from optuna.samplers import TPESampler


from fueling.learning.autotuner.tuner.optuna_optimization_tuner import OptunaOptimizationTuner
import fueling.common.logging as logging

flags.DEFINE_string(
    "study_storage_url",
    "postgres:5432",
    "URL to access a relational database"
)
flags.DEFINE_string(
    "study_name",
    "",
    "study name for optuna, this is necessary if running optuna in parallel. Otherwise, generate a random name."
)
flags.DEFINE_integer(
    "n_coworkers",
    1,
    "number of optuna workers running in parallel (including itself)"
)


class OptunaSuperTuner(OptunaOptimizationTuner):
    """Super Optuna tuner that spawns multiple tuners on a distributed system"""

    def init_optimizer_visualizer(self, tuner_parameters):
        study_name = flags.FLAGS.study_name or f"autotuner-{tuner.timestamp}"

        overall_iterations = self.tuner_param_config_pb.tuner_parameters.n_iter
        self.n_iter = math.ceil(overall_iterations / flags.FLAGS.n_coworkers)

        logging.info(f"Running {study_name} for {self.n_iter} trials...")
        self.visualizer = optuna.visualization
        self.optimizer = optuna.create_study(
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


if __name__ == "__main__":
    tic_start = time.perf_counter()

    flags.FLAGS(sys.argv)
    tuner = OptunaSuperTuner()
    tuner.run()

    logging.info(f"Timer: overall optuna tuning  - {time.perf_counter() - tic_start: 0.04f} sec")
