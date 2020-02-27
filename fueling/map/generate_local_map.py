#!/usr/bin/env python
"""
This is a module to gen local map
"""

from datetime import datetime
import os
import glob

from absl import flags
import pyspark_utils.helper as spark_helper

from fueling.common.partners import partners
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.email_utils as email_utils
import fueling.common.redis_utils as redis_utils

flags.DEFINE_integer('zone_id', 50, 'the zone id of local.')
flags.DEFINE_string('lidar_type', 'lidar16', 'compensator pointcloud topic.')


class LocalMapPipeline(BasePipeline):
    """generate local map"""

    def run_test(self):
        """Local mini test."""
        dir_prefix = '/apollo/data/bag'
        src_prefix = os.path.join(dir_prefix, 'data')
        dst_prefix = os.path.join(dir_prefix, 'result')
        zone_id = self.FLAGS.get('zone_id')
        lidar_type = self.FLAGS.get('lidar_type')
        if not os.path.exists(dst_prefix):
            logging.warning('src_prefix path: {} not exists'.format(dst_prefix))
            file_utils.makedirs(dst_prefix)
        else:
            logging.info("target_prefix: {}".format(dst_prefix))
        # RDD(record_path)
        todo_records = self.to_rdd([src_prefix])
        self.run(todo_records, src_prefix, dst_prefix, zone_id, lidar_type)
        logging.info('local map gen: Done, TEST')

    def run_prod(self):
        """Production."""
        src_prefix = self.FLAGS.get('input_data_path', 'test/virtual_lane/data')
        dst_prefix = self.FLAGS.get('output_data_path', 'test/virtual_lane/result')
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        zone_id = self.FLAGS.get('zone_id')
        lidar_type = self.FLAGS.get('lidar_type')
        logging.info("job_id: %s" % job_id)

        object_storage = self.partner_storage() or self.our_storage()
        source_dir = object_storage.abs_path(src_prefix)
        logging.info('source_dir path is {}'.format(source_dir))

        target_prefix = os.path.join(dst_prefix, job_owner, job_id)
        target_dir = object_storage.abs_path(target_prefix)
        if not os.path.exists(target_dir):
            logging.warning('bos path: {} not exists'.format(target_dir))
            file_utils.makedirs(target_dir)
        else:
            logging.info("target_dir: {}".format(target_dir))
        path = '/apollo/bazel-bin/modules/localization/msf/local_tool/data_extraction/compare_poses'
        if not os.path.exists(path):
            logging.warning('compare_poses: {} not exists'.format(path))

        receivers = email_utils.SIMPlEHDMAP_TEAM + email_utils.D_KIT_TEAM
        partner = partners.get(job_owner)
        if partner:
            receivers.append(partner.email)
        title = 'Your virtual lane is generated!'
        content = {'Job Owner': job_owner, 'Job ID': job_id}

        velodyne16_ext_list = glob.glob(os.path.join(source_dir, '*.yaml'))
        logging.info('velodyne16_ext_list: {}'.format(velodyne16_ext_list))

        job_type = 'VIRTUAL_LANE_GENERATION'
        redis_key = F'External_Partner_Job.{job_owner}.{job_type}.{job_id}'

        if not velodyne16_ext_list:
            logging.error('velodyne16_novatel_extrinsics_example.yaml not exists')
            title = 'Your localmap is not generated!'
            email_utils.send_email_info(title, content, receivers)
            redis_value = {'end_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                           'job_status': 'Local map failed'}
            redis_utils.redis_extend_dict(redis_key, redis_value)
            return

        # RDD(tasks), the tasks without source_dir as prefix
        # RDD(record_path)
        todo_records = self.to_rdd([source_dir])
        self.run(todo_records, source_dir, target_dir, zone_id, lidar_type)

        email_utils.send_email_info(title, content, receivers)

        redis_value = {'end_time': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'),
                       'job_status': 'success'}
        redis_utils.redis_extend_dict(redis_key, redis_value)

    def run(self, todo_records, src_prefix, dst_prefix, zone_id, lidar_type):
        """Run the pipeline with given arguments."""
        # Spark cascade style programming.
        self.dst_prefix = dst_prefix
        self.zone_id = zone_id
        self.lidar_type = lidar_type
        spark_helper.cache_and_log('gen_local_map',
                                   # RDD(source_dir)
                                   todo_records
                                   # RDD(map)
                                   .map(self.execute_task))

    def execute_task(self, source_dir):
        """Execute task by task"""
        logging.info('executing task with src_dir: {}'.format(source_dir))

        local_map_creator_bin = 'bash /apollo/scripts/msf_simple_map_creator.sh'
        # msf_simple_map_creator.sh [records folder] [extrinsic_file] [zone_id]
        # [map folder] [lidar_type]
        velodyne16_ext_list = glob.glob(os.path.join(source_dir, '*.yaml'))
        local_command = '{} {} {} {} {} {}'.format(
            local_map_creator_bin, source_dir, velodyne16_ext_list[0],
            self.zone_id, self.dst_prefix, self.lidar_type)
        logging.info('local_map_creator command is {}'.format(local_command))
        return_code = os.system(local_command)
        if return_code == 0:
            logging.info('Successed to generate local map')
        else:
            logging.error('Failed to generate local map')


if __name__ == '__main__':
    LocalMapPipeline().main()
