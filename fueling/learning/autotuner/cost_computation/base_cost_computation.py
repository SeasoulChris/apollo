#!/usr/bin/env python
"""Auto Tuner Base"""

# standard packages
import csv
import json
import uuid
import os
import time

# third party packages
from absl import flags
import pyspark_utils.op as spark_op

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
from fueling.learning.autotuner.client.sim_client import SimClient
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging

# Flags
flags.DEFINE_string("training_id", None, "A unique id")
flags.DEFINE_string("commit_id", None, "Apollo commit id.")

flags.DEFINE_string(
    "scenario_path",
    "fueling/learning/autotuner/config/sim_scenarios.csv",
    "File path to list of scenarios in csv format.",
)
flags.DEFINE_string(
    "mnt_root_dir", "/mnt/bos", "BOS directory"
)
flags.DEFINE_string(
    "record_output_dir",
    "autotuner",
    "The relative path to the BOS directory that stores output record files from simulation",
)
flags.DEFINE_string(
    "sim_service_url", "localhost:50051", "channel url to sim service"
)


class BaseCostComputation(BasePipeline):
    def init(self):
        BasePipeline.init(self)

        mode = "CLOUD" if self.FLAGS.get('running_mode') == 'PROD' else "LOCAL"
        logging.info(f"Running cost_computation in {mode} mode.")

        if not flags.FLAGS.commit_id:
            logging.error("Apollo commit id not specified.")
            return False

        if not flags.FLAGS.training_id:
            logging.error("Training id not specified.")
            return False

        return True

    def run(self):
        self.run_once()

    def run_once(self):
        if not self.init():
            return

        # build
        if not self.build():
            return

        # RDD(scenario_id)
        scenario_id_rdd = self.get_scenarios()

        # RDD(config_id)
        self.config_id_2_pb2 = self.get_config_map()
        config_id_rdd = self.to_rdd(self.config_id_2_pb2)

        config_2_score = (
            # RDD(config_id)
            config_id_rdd
            # RDD((config_id, scenario_id))
            .cartesian(scenario_id_rdd)
            # PairRDD((config_id, scenario_id), bag_path)
            .map(spark_op.value_by(self.run_scenario))
            # PairRDD((config_id, scenario_id), score)
            .mapValues(self.calculate_individual_score)
            # PairRDD(config_id, [((config_id, scenario_id), score)])
            .groupBy(self.group_by_config_id)
            # PairRDD(config_id, weighted_score)
            .mapValues(self.calculate_weighted_score)
        ).collect()

        # config_id -> weight_score
        self.save_weighted_score(config_2_score)

    def set_sim_channel(self):
        url = self.FLAGS.get('sim_service_url')
        logging.info(f'Setting sim service url to {url}')
        SimClient.set_channel(url)

    def build(self):
        self.set_sim_channel()
        return SimClient.trigger_build(self.FLAGS.get("commit_id"))

    def get_scenarios(self):
        """Return RDD(scenario_id)"""
        scenario_path = file_utils.fuel_path(self.FLAGS.get("scenario_path"))
        with open(scenario_path) as scenario_file:
            reader = csv.reader(scenario_file, delimiter=",")
            scenarios = []
            for line in reader:
                # skip comment
                if len(line) > 0 and line[0].startswith("#"):
                    continue
                scenarios.extend([int(id) for id in line if id.isnumeric()])

        logging.info(f"Training scenarios are {scenarios}")

        return self.to_rdd(scenarios)

    def get_config_map(self):
        """Return a map of map: {config_id: {local_config_file_path: serialized_config}} """
        raise Exception("Not implemented!")

    def run_scenario(self, input):
        """Trigger Simulation with the given configuration and scenario"""
        (config_id, scenario_id) = input
        logging.info(f"Setting up scenario {scenario_id} with config id {config_id}")

        training_id = self.FLAGS.get("training_id")
        job_id = uuid.uuid4().hex
        record_relative_dir = f"{self.FLAGS.get('record_output_dir')}/{training_id}/{job_id}"
        record_filename = f"{config_id}_{scenario_id}.record"

        self.set_sim_channel()
        success = SimClient.run_scenario(
            training_id,
            self.FLAGS.get("commit_id"),
            scenario_id,
            self.config_id_2_pb2[config_id],
            record_relative_dir,
            record_filename,
        )
        if not success:
            raise Exception(f"Failed to run scenario {scenario_id}")

        record_absolute_dir = f"{self.get_temp_dir()}/{job_id}"
        # Retry in case the BOS dir mounted has network delay
        for t in range(1, 6):
            if os.path.exists(f"{record_absolute_dir}/{record_filename}"):
                logging.info(f"Found result bag {record_filename}")
                break
            elif t == 5:
                raise Exception(f"No bag found after running scenario: {record_absolute_dir}")
            else:
                logging.info(f"Retry fetching result bag in {t}min...")
                time.sleep(60 * t)  # sleep time increases from 1min, to 4min

        return record_absolute_dir

    def calculate_individual_score(self, bag_path):
        """Return score(s) from the given Cyber record"""
        raise Exception("Not implemented!")

    def group_by_config_id(self, input):
        (key, bag_path) = input
        (config_id, _) = key
        return config_id

    def calculate_weighted_score(self, config_2_score):
        """Return a weighted score from (config, scenario) -> score(s) map"""
        raise Exception("Not implemented!")

    def get_temp_dir(self):
        return f"{self.FLAGS.get('mnt_root_dir')}/{self.FLAGS.get('record_output_dir')}/{self.FLAGS.get('training_id')}"

    def save_weighted_score(self, score):
        with open(f"{self.get_temp_dir()}/scores.out", "w") as output_score_file:
            output_score_file.write(json.dumps(dict(score)))
