#!/usr/bin/env python
"""Auto Tuner Base"""

# standard packages
import time

# third party packages
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class BaseAutoTuner(BasePipeline):
    def __init__(self):
        self.iter_count = 0

    def run_test(self):
        self.run()

    def run_prod(self):
        self.run()

    def run(self):
        # build
        self.build()

        # RDD(scenarios)
        scenarios = self.get_scenarios()

        while not self.is_done():
            logging.info(f'==== Iteration {self.iter_count} ====')
            self.iter_count += 1

            # PairRDD((configuration, scenario), score)
            config_2_score = spark_helper.cache_and_log(
                'config_2_score',
                # RDD(configuration)
                self.generate_config()
                # RDD((configuration, scenario))
                .cartesian(scenarios)
                # PairRDD((configuration, scenario), bag_path)
                .map(spark_op.value_by(self.run_scenario))
                # PairRDD((configuration, scenario), score)
                .map(self.calculate_score),
                1
            )

            self.train_and_resample(config_2_score)

    def build(self):
        # TODO(vivian): trigger sim build
        logging.info("Compiling replay engine...")

    def get_scenarios(self):
        # TODO(vivian): read scenarios from a config file
        return self.to_rdd(range(1))

    def generate_config(self):
        raise Exception("Not implemented!")

    def run_scenario(self, input):
        """
            Trigger Simulation with the given configuration and scenario,
            and return bag path.
        """
        # TODO(vivian): send request to simulation
        logging.info(f'Running scenario with (config, scenario): {input}')
        time.sleep(input[1])
        return f'rosbag-{"-".join([str(item) for item in input])}'

    def calculate_score(self, input):
        raise Exception("Not implemented!")

    def train_and_resample(self, dataset):
        raise Exception("Not implemented!")

    def is_done(self):
        raise Exception("Not implemented!")
