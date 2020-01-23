#!/usr/bin/env python
"""Auto Tuner Base"""

# standard packages
import csv
import uuid

# third party packages
from absl import flags
import pyspark_utils.op as spark_op

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
from fueling.autotuner.client.sim_client import SimClient
import fueling.common.logging as logging

# Flags
flags.DEFINE_string("training_id", None, "A unique id")
flags.DEFINE_string("commit", None, "Apollo Commit id.")
flags.DEFINE_string(
    "scenario_path",
    "./fueling/autotuner/config/sim_scenarios.csv",
    "File path to list of scenarios in csv format.",
)
flags.DEFINE_string(
    "record_output_dir",
    "/apollo/autotuner/bags",
    "The BOS directory that stores output record files from simulation",
)

TMP_ROOT_DIR = "/tmp/autotuner"


class BaseCostComputation(BasePipeline):
    def init(self):
        if not flags.FLAGS.commit:
            logging.error("Apollo commit id not specified.")
            return False

        if not flags.FLAGS.training_id:
            logging.error("Training id not specified.")
            return False

        self.FLAGS = flags.FLAGS.flag_values_dict()
        return True

    def run_test(self):
        self.run_once()

    def run_prod(self):
        self.run_once()

    def run_once(self):
        if not self.init():
            return

        # build
        if not self.build():
            return

        # RDD(scenarios)
        scenarios_rdd = self.get_scenarios()

        config_2_score = (
            # RDD(configuration)
            self.generate_config_rdd()
            # RDD((configuration, scenario))
            .cartesian(scenarios_rdd)
            # PairRDD((configuration, scenario), bag_path)
            .map(spark_op.value_by(self.run_scenario))
            # PairRDD((configuration, scenario), score)
            .map(self.calculate_individual_score)
        ).collect()

        score = self.calculate_weighted_score(config_2_score)
        self.save_weighted_score(score)

    def build(self):
        return SimClient.trigger_build(self.FLAGS.get("commit"))

    def get_scenarios(self):
        """Return RDD(scenario_id)"""
        with open(self.FLAGS.get("scenario_path")) as scenario_file:
            reader = csv.reader(scenario_file, delimiter=",")
            scenarios = []
            for line in reader:
                # skip comment
                if len(line) > 0 and line[0].startswith("#"):
                    continue
                scenarios.extend([int(id) for id in line if id.isnumeric()])

        logging.info(f"Training scenarios are {scenarios}")

        return self.to_rdd(scenarios)

    def generate_config_rdd(self):
        """Return RDD({local_config_file_path: serialized_config})"""
        raise Exception("Not implemented!")

    def run_scenario(self, input):
        """Trigger Simulation with the given configuration and scenario"""
        (config, scenario) = input
        training_id = self.FLAGS.get("training_id")
        job_id = uuid.uuid1().hex
        output_path = f"{self.FLAGS.get('record_output_dir')}/{training_id}"
        output_file = f"{job_id}.record"

        # TODO: handle error status
        status = SimClient.run_scenario(
            training_id,
            self.FLAGS.get("commit"),
            scenario,
            config,
            output_path,
            output_file,
        )

        return f"{output_path}/{output_file}"

    def calculate_individual_score(self, input):
        """Return score(s) from the given Cyber record"""
        raise Exception("Not implemented!")

    def calculate_weighted_score(self, config_2_score):
        """Return a weighted score from (config, scenario) -> score(s) map"""
        raise Exception("Not implemented!")

    def get_temp_dir(self):
        return f"{TMP_ROOT_DIR}/{self.FLAGS.get('training_id')}"

    def save_weighted_score(self, score):
        with open(f"{self.get_temp_dir()}/score.out", "w") as output_file:
            output_file.write(str(score))
