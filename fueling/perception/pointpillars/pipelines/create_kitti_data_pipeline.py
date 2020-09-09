#!/usr/bin/env python

import time
import cv2 as cv

from fueling.common.base_pipeline import BasePipeline
from fueling.common.job_utils import JobUtils
from fueling.perception.pointpillars.second.create_data import kitti_data_prep
import fueling.common.context_utils as context_utils
import fueling.common.logging as logging


class CreateDataKitti(BasePipeline):

    def run(self):
        """Run."""
        job_id = self.FLAGS.get('job_id')
        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.create_data_kitti)
        logging.info('create data complete in {} seconds.'.format(time.time() - time_start))
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_progress(20)

    def create_data_kitti(self, instance_id):
        """Run create data task"""
        cv.setNumThreads(0)

        input_data_path = self.FLAGS.get('input_data_path')
        object_storage = self.partner_storage() or self.our_storage()
        data_path = object_storage.abs_path(input_data_path)

        kitti_data_prep(data_path)


if __name__ == '__main__':
    CreateDataKitti().main()
