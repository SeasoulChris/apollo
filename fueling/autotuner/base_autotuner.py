#!/usr/bin/env python
"""Auto Tuner Base"""

# standard packages
import csv
import uuid

# third party packages
from absl import flags
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
from fueling.autotuner.client.sim_client import SimClient
import fueling.common.logging as logging

# Flags
flags.DEFINE_string("commit", None, "Apollo Commit id.")
flags.DEFINE_string(
    "scenario_path",
    "./fueling/autotuner/config/sim_scenarios.csv",
    "File path to list of scenarios in csv format.",
)
flags.DEFINE_string(
    "record_output_dir",
    "/autotuner/bags",
    "The BOS directory that stores output record files from simulation",
)


class BaseAutoTuner(BasePipeline):
    def __init__(self):
        self.training_id = uuid.uuid1().hex
        self.iter_count = 0

    def init(self):
        # Member variables are available on both driver and executors.
        self.FLAGS = flags.FLAGS.flag_values_dict()

        if not flags.FLAGS.commit:
            logging.error("Apollo commit id not specified.")
            return False

        return True

    def run_test(self):
        self.run()

    def run_prod(self):
        self.run()

    def run(self):
        if not self.init():
            return

        # build
        if not self.build():
            logging.error("Failed to build replay engine.")
            return

        # RDD(scenarios)
        scenarios_rdd = self.get_scenarios()

        while not self.is_done():
            logging.info(f"==== Iteration {self.iter_count} ====")
            self.iter_count += 1

            # PairRDD((configuration, scenario), score)
            config_2_score = spark_helper.cache_and_log(
                "config_2_score",
                # RDD(configuration)
                self.generate_config()
                # RDD((configuration, scenario))
                .cartesian(scenarios_rdd)
                # PairRDD((configuration, scenario), bag_path)
                .map(spark_op.value_by(self.run_scenario))
                # PairRDD((configuration, scenario), score)
                .map(self.calculate_score),
                1,
            )

            self.train_and_resample(config_2_score)

    def build(self):
        return SimClient.trigger_build(flags.FLAGS.commit)

    def get_scenarios(self):
        """Return RDD(scenario_id)"""
        with open(flags.FLAGS.scenario_path) as scenario_file:
            reader = csv.reader(scenario_file, delimiter=",")
            scenarios = []
            for line in reader:
                # skip comment
                if len(line) > 0 and line[0].startswith("#"):
                    continue
                scenarios.extend([int(id) for id in line if id.isnumeric()])

        logging.info(f"Training scenarios are {scenarios}")

        return self.to_rdd(scenarios)

    def generate_config(self):
        """Return PairRDD(local_config_file_path, serialized_config)"""
        raise Exception("Not implemented!")

    def run_scenario(self, input):
        """Trigger Simulation with the given configuration and scenario"""
        (config, scenario) = input
        logging.info(f"Running scenario {scenario} for {config} ...")

        job_id = uuid.uuid1().hex
        output_path = f"{self.FLAGS.get('record_output_dir')}/{self.training_id}"
        output_file = f"{job_id}.record"

        status = SimClient.run_scenario(
            self.training_id,
            self.FLAGS.get("commit"),
            scenario,
            config,
            output_path,
            output_file,
        )

        return f"{output_path}/{output_file}"

    def calculate_score(self, input):
        raise Exception("Not implemented!")

    def train_and_resample(self, dataset):
        raise Exception("Not implemented!")

    def is_done(self):
        raise Exception("Not implemented!")
