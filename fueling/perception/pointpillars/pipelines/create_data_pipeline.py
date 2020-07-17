#!/usr/bin/env python

import os
import sys
import time

import cv2 as cv

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
from fueling.perception.pointpillars.second.create_data import nuscenes_data_prep


class CreateDataNuscenes(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.create_data_nuscenes)
        logging.info('create data complete in {} seconds.'.format(time.time() - time_start))

    @staticmethod
    def create_data_nuscenes(instance_id):
        """Run create data task"""
        cv.setNumThreads(0)

        root_path = "/data/perception_data/nuscenes_data"
        version = "v1.0-trainval"
        max_sweeps = 10
        dataset_name = "NuScenesDataset"

        nuscenes_data_prep(root_path, version, dataset_name, max_sweeps)


if __name__ == '__main__':
    CreateDataNuscenes().main()
