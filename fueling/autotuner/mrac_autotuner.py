#!/usr/bin/env python

from fueling.autotuner.base_autotuner import BaseAutoTuner
import fueling.common.logging as logging


class MracAutoTuner(BaseAutoTuner):
    def __init__(self, iter_num):
        BaseAutoTuner.__init__(self)
        self.iter_num = iter_num

    def generate_config(self):
        # TODO: implement me
        logging.info("Generating config...")
        return self.to_rdd(
            [
                {"/apollo/path/to/config": "c1"},
                {"/apollo/path/to/config": "c2"},
                {"/apollo/path/to/config": "c3"},
            ]
        )

    def calculate_score(self, input):
        # TODO: implement me
        logging.info(f"Calculating score for: {input}")
        (key, bag_path) = input
        return (key, f"SCORE:{bag_path}")

    def train_and_resample(self, dataset):
        # TODO: implement me
        logging.info("Training...")
        return "training results"

    def is_done(self):
        return self.iter_count >= self.iter_num


if __name__ == "__main__":
    MracAutoTuner(2).main()
