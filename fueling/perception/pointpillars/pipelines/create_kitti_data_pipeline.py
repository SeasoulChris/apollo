#!/usr/bin/env python

import time
import cv2 as cv
import os

from fueling.common.base_pipeline import BasePipeline
from fueling.common.job_utils import JobUtils
from fueling.perception.pointpillars.second.create_data import kitti_data_prep
import fueling.common.context_utils as context_utils
import fueling.common.logging as logging


class CreateDataKitti(BasePipeline):

    def run_test(self):
        data_path = '/fuel/testdata/perception/pointpillars/kitti_testdata/'
        kitti_data_prep(data_path)

    def run(self):
        """Run."""
        self.if_error = False
        job_id = self.FLAGS.get('job_id')
        input_data_path = self.FLAGS.get('input_data_path')
        object_storage = self.partner_storage() or self.our_storage()
        self.input_data_path = object_storage.abs_path(input_data_path)
        check_flag, error_text = self.input_path_check(self.input_data_path)
        if not check_flag:
            if self.if_error:
                JobUtils(job_id).save_job_failure_code('E700')
                JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                     error_text, False)
                return

        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.create_data_kitti)
        logging.info('create data complete in {} seconds.'.format(time.time() - time_start))
        if context_utils.is_cloud():
            JobUtils(job_id).save_job_progress(20)
            if self.if_error:
                error_text = "Failed to create data for preprocessing"
                JobUtils(job_id).save_job_failure_code('E701')
                JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                     error_text, False)

    def create_data_kitti(self, instance_id):
        """Run create data task"""
        cv.setNumThreads(0)

        data_path = self.input_data_path
        try:
            kitti_data_prep(data_path)
        except BaseException:
            logging.error('Failed to create data')
            self.if_error = True

    def input_path_check(self, input_folder):
        input_dir_list = [subdir for subdir in os.listdir(input_folder)
                          if os.path.isdir(os.path.join(input_folder, subdir))]
        input_file_list = [subdir for subdir in os.listdir(input_folder)
                           if os.path.isfile(os.path.join(input_folder, subdir))]

        if 'training' not in input_dir_list or 'testing' not in input_dir_list:
            logging.error(' Input missing training or testing file folder ')
            error_text = ' Input missing training or testing file folder '
            self.if_error = True
            return False, error_text
        if 'train.txt' not in input_file_list or 'val.txt' not in input_file_list:
            logging.error(' Input missing train.txt or val.txt file ')
            error_text = ' Input missing train.txt or val.txt file '
            self.if_error = True
            return False, error_text

        need_input_names = ['calib', 'velodyne', 'label_2', 'image_2']

        for input_dir in ['training', 'testing']:
            input_subdir = os.path.join(input_folder, input_dir)
            input_subdir_list = [subdir for subdir in os.listdir(input_subdir)
                                 if os.path.isdir(os.path.join(input_subdir, subdir))]
            for need_input_name in need_input_names:
                if need_input_name not in input_subdir_list:
                    logging.error(' Input missing {} file folder'.format(need_input_name))
                    error_text = ' Input missing {} file folder'.format(need_input_name)
                    self.if_error = True
                    return False, error_text
        return True, ""


if __name__ == '__main__':
    CreateDataKitti().main()
