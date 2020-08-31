#!/usr/bin/env python

import time
from absl import flags
import cv2 as cv

from fueling.common.base_pipeline import BasePipeline
from fueling.perception.pointpillars.second.create_data import nuscenes_data_prep
import fueling.common.logging as logging

flags.DEFINE_string('input_data_path', '/mnt/bos/modules/perception/pointpillars/data/',
                    'training data path')


class CreateDataNuscenes(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.create_data_nuscenes)
        logging.info('create data complete in {} seconds.'.format(time.time() - time_start))

    def create_data_nuscenes(self, instance_id):
        """Run create data task"""
        cv.setNumThreads(0)

        root_path = self.FLAGS.get('input_data_path')
        version = "v1.0-mini"
        max_sweeps = 10
        dataset_name = "NuScenesDataset"

        nuscenes_data_prep(root_path, version, dataset_name, max_sweeps)


if __name__ == '__main__':
    CreateDataNuscenes().main()
