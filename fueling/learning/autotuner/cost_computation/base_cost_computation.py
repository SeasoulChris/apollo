#!/usr/bin/env python
"""Auto Tuner Base"""

import json
import uuid
import os
import time

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.learning.autotuner.client.sim_client import SimClient
from fueling.learning.autotuner.common.utils import run_with_retry
import fueling.common.context_utils as context_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.spark_op as spark_op
import fueling.learning.autotuner.proto.sim_service_pb2 as sim_service_pb2
import fueling.learning.autotuner.proto.cost_computation_service_pb2 as cost_service_pb2


flags.DEFINE_string("token", None, "Sim service token.")
flags.DEFINE_string("iteration_id", None, "A unique id")
flags.DEFINE_string("mnt_root_dir", "/mnt/bos", "BOS directory")
flags.DEFINE_string(
    "record_output_dir", "autotuner",
    "The relative path to the BOS directory that stores output record files from simulation")
flags.DEFINE_string("sim_service_url", "localhost:50051", "channel url to sim service")


class BaseCostComputation(BasePipeline):
    def init(self):
        BasePipeline.init(self)

        logging.info(f"Running cost_computation in {self.FLAGS.get('running_mode')} mode.")

        if not flags.FLAGS.token:
            logging.error("Service token not specified.")
            return False

        if not flags.FLAGS.iteration_id:
            logging.error("Iteration id not specified.")
            return False

        return True

    def run(self):
        self.run_once()

    def run_once(self):
        if not self.init():
            return
        tic_start = time.perf_counter()

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
        logging.info(f"Timer: total run_once - {time.perf_counter() - tic_start:0.04f} sec")

    def set_sim_channel(self):
        url = self.FLAGS.get('sim_service_url')
        logging.info(f'Setting sim service url to {url}')
        SimClient.set_channel(url)

    def pause_to_debug(self):
        if not context_utils.is_cloud():
            time.sleep(600)  # keep the exec pod for some time if error

    def get_scenarios(self):
        """Return RDD(scenario_id)"""
        request_pb2 = cost_service_pb2.InitRequest()
        proto_utils.get_pb_from_text_file(
            f"{self.get_absolute_training_dir()}/init_request.pb.txt", request_pb2,
        )
        logging.info(f"Training scenarios are {request_pb2.scenario_id}")
        return self.to_rdd(request_pb2.scenario_id)

    def get_dynamic_model(self):
        """Return dynamic model enum"""
        return sim_service_pb2.DynamicModel.ECHO_LINCOLN

    def get_config_map(self):
        """Return a map of map: {config_id: {local_config_file_path: serialized_config}} """
        raise Exception("Not implemented!")

    def run_scenario(self, input):
        """Trigger Simulation with the given configuration and scenario"""
        tic_start = time.perf_counter()
        (config_id, scenario_id) = input
        logging.info(f"Setting up scenario {scenario_id} with config id {config_id}")

        job_id = f"{config_id}_{scenario_id}"
        record_relative_dir = f"{self.get_relative_iter_dir()}/{job_id}"
        record_filename = f"{job_id}.record"

        self.set_sim_channel()
        status = run_with_retry(
            3,  # max retries
            SimClient.run_scenario,
            self.FLAGS.get("token"),
            self.FLAGS.get("iteration_id"),
            scenario_id,
            self.config_id_2_pb2[config_id],
            record_relative_dir,
            record_filename,
        )

        if status.message.startswith('finish'):
            logging.info(f"Done running scenario {scenario_id} for {record_filename}.")
        else:
            self.pause_to_debug()
            raise Exception(f"Failed to run scenario {scenario_id}: {status}")

        record_absolute_dir = f"{self.get_absolute_iter_dir()}/{job_id}"
        if not file_utils.file_exists(f"{record_absolute_dir}/{record_filename}"):
            raise Exception(f"No bag found after running scenario: {record_absolute_dir}")

        logging.info(f"Timer: total run_scenario - {time.perf_counter() - tic_start:0.04f} sec")
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

    def get_relative_training_dir(self):
        return f"{self.FLAGS.get('record_output_dir')}/{self.FLAGS.get('token')}"

    def get_relative_iter_dir(self):
        return f"{self.get_relative_training_dir()}/{self.FLAGS.get('iteration_id')}"

    def get_absolute_training_dir(self):
        return f"{self.FLAGS.get('mnt_root_dir')}/{self.get_relative_training_dir()}"

    def get_absolute_iter_dir(self):
        return f"{self.FLAGS.get('mnt_root_dir')}/{self.get_relative_iter_dir()}"

    def save_weighted_score(self, score):
        with open(f"{self.get_absolute_iter_dir()}/scores.out", "w") as output_score_file:
            output_score_file.write(json.dumps(dict(score)))
