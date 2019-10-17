#!/usr/bin/env python
"""
This is a module to gen local map
"""

import os
import glob

from absl import flags
import pyspark_utils.helper as spark_helper

import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient

flags.DEFINE_string('input_data_path', 'simplehdmap',
                    'simple hdmap input/output data path.')

class LocalMapPipeline(BasePipeline):

    """generate local map"""

    def __init__(self):
        """Initialize"""
        BasePipeline.__init__(self, 'local_map')

    def run_test(self):
        """Local mini test."""
        dir_prefix = '/apollo/data/bag'
        src_prefix = os.path.join(dir_prefix, 'data')
        dst_prefix = os.path.join(dir_prefix, 'result')
        if not os.path.exists(dst_prefix):
            logging.warning('src_prefix path: {} not exists'.format(dst_prefix))
            file_utils.makedirs(dst_prefix)
        else:
            logging.info("target_prefix: {}".format(dst_prefix))
        # RDD(record_path)
        todo_records = self.to_rdd([src_prefix])
        self.run(todo_records, src_prefix, dst_prefix)
        logging.info('local map gen: Done, TEST')

    def run_prod(self):
        """Production."""
        dir_prefix = self.FLAGS.get('input_data_path')
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        logging.info("job_id: %s" % job_id)

        #src_prefix = 'simplehdmap/result'
        src_prefix = os.path.join(dir_prefix, 'data')
        dst_prefix = os.path.join(dir_prefix, 'result')        

        bos_client = BosClient()
        object_storage = self.partner_object_storage() or bos_client
        source_dir = object_storage.abs_path(src_prefix)
        logging.info('source_dir path is {}'.format(source_dir))

        target_prefix = os.path.join(dst_prefix, job_owner, job_id)
        target_dir = object_storage.abs_path(target_prefix)
        if not os.path.exists(target_dir):
            logging.warning('bos path: {} not exists'.format(target_dir))
            file_utils.makedirs(target_dir)
        else:
            logging.info("target_dir: {}".format(target_dir))        

        velodyne16_ext_path = os.path.join(source_dir, 'velodyne16_novatel_extrinsics_example.yaml')
        if not os.path.exists(velodyne16_ext_path):
            logging.warning('velodyne16_novatel_extrinsics_example.yaml: {} not exists'.format(velodyne16_ext_path))
               
        # RDD(tasks), the tasks without source_dir as prefix
        # RDD(record_path)
        todo_records = self.to_rdd([source_dir])
        self.run(todo_records, source_dir, target_dir)       

    def run(self, todo_records, src_prefix, dst_prefix):
        """Run the pipeline with given arguments."""
        # Spark cascade style programming.
        self.dst_prefix = dst_prefix
        record_points = spark_helper.cache_and_log('gen_local_map',
                                                   # RDD(source_dir)
                                                   todo_records
                                                   # RDD(map)
                                                   .map(self.execute_task))

    def execute_task(self, source_dir):
        """Execute task by task"""
        logging.info('executing task with src_dir: {}'.format(source_dir))

        local_map_creator_bin = 'bash /apollo/scripts/msf_simple_map_creator.sh'
        #msf_simple_map_creator.sh [records folder] [extrinsic_file] [zone_id] [map folder] [lidar_type]
        map_dir = self.dst_prefix
        velodyne16_ext_path = os.path.join(source_dir, 'velodyne16_novatel_extrinsics_example.yaml')
        local_command = '{} {} {} 50 {} 16'.format(
            local_map_creator_bin, source_dir, velodyne16_ext_path, map_dir)
        logging.info('local_map_creator command is {}'.format(local_command))
        return_code = os.system(local_command)
        logging.info("return code for local_map_gen is {}".format(return_code))

        if return_code != 0:
            logging.error('failed to generate local map')
            return
        logging.info('Successed to generate local map')

if __name__ == '__main__':
    LocalMapPipeline().main()
