#!/usr/bin/env python
import datetime
import operator
import os

import glog
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class GenerateSmallRecordsPipeline(BasePipeline):
    """GenerateSmallRecords pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'generate-small-records')

    def run_test(self):
        """Run test."""
        sc = self.get_spark_context()
        records_rdd = sc.parallelize(['/apollo/docs/demo_guide/demo_3.5.record'])
        whitelist_dirs_rdd = sc.parallelize(['/apollo/docs/demo_guide'])
        blacklist_dirs_rdd = sc.parallelize([])
        origin_prefix = '/apollo/docs/demo_guide'
        target_prefix = '/apollo/data'
        channels = {
            '/apollo/canbus/chassis',
            '/apollo/canbus/chassis_detail',
            '/apollo/control',
            '/apollo/control/pad',
        }
        self.run(records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
                 origin_prefix, target_prefix, channels)
    
    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        # Original records are public-test/path/to/*.record, sharded to M.
        origin_prefix = 'public-test/2019/'
        # We will process them to small-records/path/to/*.record, sharded to N.
        target_prefix = 'small-records/2019/'
        channels = {
            '/apollo/canbus/chassis',
            '/apollo/canbus/chassis_detail',
            '/apollo/control',
            '/apollo/control/pad',
            '/apollo/drive_event',
            '/apollo/guardian',
            '/apollo/localization/pose',
            '/apollo/localization/msf_gnss',
            '/apollo/localization/msf_lidar',
            '/apollo/localization/msf_status',
            '/apollo/hmi/status',
            '/apollo/monitor',
            '/apollo/monitor/system_status',
            '/apollo/navigation',
            '/apollo/perception/obstacles',
            '/apollo/perception/traffic_light',
            '/apollo/planning',
            '/apollo/prediction',
            '/apollo/relative_map',
            '/apollo/routing_request',
            '/apollo/routing_response',
            '/apollo/sensor/conti_radar',
            '/apollo/sensor/delphi_esr',
            '/apollo/sensor/gnss/best_pose',
            '/apollo/sensor/gnss/corrected_imu',
            '/apollo/sensor/gnss/gnss_status',
            '/apollo/sensor/gnss/imu',
            '/apollo/sensor/gnss/ins_stat',
            '/apollo/sensor/gnss/odometry',
            '/apollo/sensor/gnss/raw_data',
            '/apollo/sensor/gnss/rtk_eph',
            '/apollo/sensor/gnss/rtk_obs',
            '/apollo/sensor/mobileye',
            '/tf',
            '/tf_static',
        }

        files = s3_utils.list_files(bucket, origin_prefix).cache()
        records_rdd = files.filter(record_utils.is_record_file)
        # task_dir, which has a 'COMPLETE' file inside.
        whitelist_dirs_rdd = (
            files
            .filter(lambda path: path.endswith('/COMPLETE'))
            .map(os.path.dirname))
        # task_dir, whose target_dir has already been generated.
        dir_blacklist_rdd = (
            s3_utils.list_dirs(bucket, target_prefix)
            .map(lambda target_dir: target_dir.replace(target_prefix, origin_prefix, 1)))

        self.run(records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
                 origin_prefix, target_prefix, channels)

    def run(self, records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
            origin_prefix, target_prefix, channels):
        """Run the pipeline with given arguments."""
        # (task_dir, record)
        records_rdd = records_rdd.keyBy(os.path.dirname)
        records_rdd = spark_op.filter_keys(records_rdd, whitelist_dirs_rdd)
        records_rdd = spark_op.substract_keys(records_rdd, blacklist_dirs_rdd)
        # (target_dir, record)
        todo_jobs = (
            records_rdd
            .map(spark_op.do_key(lambda src_dir: src_dir.replace(origin_prefix, target_prefix, 1)))
            .cache())

        # Read the input data and write to target file.
        records_count = (todo_jobs
            .flatMapValues(record_utils.read_record(channels))
                                                                   # -> (target_dir, PyBagMessage)
            .map(GenerateSmallRecordsPipeline.shard_to_file)       # -> (target_file, PyBagMessage)
            .groupByKey()                                          # -> (target_file, PyBagMessages)
            .mapValues(lambda msgs: sorted(msgs, key=lambda msg: msg.timestamp))
                                             # -> (target_file, PyBagMessages_sequence)
            .map(record_utils.write_record)  # -> (None)
            .count())                        # Simply trigger action.
        glog.info('Finished %d records!' % records_count)

        # Create COMPLETE mark.
        tasks_count = (todo_jobs
            .keys()                                            # -> target_dir
            .distinct()                                        # -> unique_target_dir
            .map(lambda path: os.path.join(s3_utils.S3_MOUNT_PATH, path, 'COMPLETE'))
                                                               # -> unique_target_dir/COMPLETE
            .map(os.mknod)                                     # Touch file
            .count())                                          # Simply trigger action.
        glog.info('Finished %d tasks!' % tasks_count)

    @staticmethod
    def shard_to_file(dir_msg):
        target_dir, msg = dir_msg
        dt = datetime.datetime.fromtimestamp(msg.timestamp / (10**9))
        target_file = os.path.join(target_dir, dt.strftime('%Y-%m-%d-%H-%M-00.record'))
        return target_file, msg


if __name__ == '__main__':
    GenerateSmallRecordsPipeline().run_test()
