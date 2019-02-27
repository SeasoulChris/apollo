#!/usr/bin/env python
import errno
import os
import pprint

import glog
import pyspark_utils.op as spark_op

from cyber_py.record import RecordReader, RecordWriter

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.time_utils as time_utils


class GenerateSmallRecords(BasePipeline):
    """GenerateSmallRecords pipeline."""
    CHANNELS = {
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
        '/apollo/routing_response_history',
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
        self.run(records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd, origin_prefix, target_prefix)
    
    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        # Original records are public-test/path/to/*.record, sharded to M.
        origin_prefix = 'public-test/2019/'
        # We will process them to small-records/path/to/*.record, sharded to N.
        target_prefix = 'small-records/2019/'

        files = s3_utils.list_files(bucket, origin_prefix).cache()
        records_rdd = files.filter(record_utils.is_record_file)
        # task_dir, which has a 'COMPLETE' file inside.
        whitelist_dirs_rdd = (
            files.filter(lambda path: path.endswith('/COMPLETE'))
            .map(os.path.dirname))
        # task_dir, whose mappinged target dir has not a 'COMPLETE' file inside.
        blacklist_dirs_rdd = (
            s3_utils.list_files(bucket, target_prefix)
            .filter(lambda path: path.endswith('/COMPLETE'))
            .map(os.path.dirname)
            .map(lambda path: path.replace(target_prefix, origin_prefix, 1)))
        self.run(records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd, origin_prefix, target_prefix)

    def run(self, records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
            origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        # (task_dir, record)
        todo_jobs = spark_op.filter_keys(records_rdd.keyBy(os.path.dirname), whitelist_dirs_rdd)
        tasks_count = (
            # -> (task_dir, record)
            spark_op.substract_keys(todo_jobs, blacklist_dirs_rdd)
            # -> (target_dir, record)
            .map(lambda dir_record: (
                s3_utils.abs_path(dir_record[0].replace(origin_prefix, target_prefix, 1)),
                s3_utils.abs_path(dir_record[1])))
            # -> (target_dir, (record, header))
            .mapValues(lambda record: (record, record_utils.read_record_header(record)))
            # -> (target_dir, (record, header)), where header is valid
            .filter(spark_op.filter_value(lambda header_record: header_record[1] is not None))
            # -> (target_file, (record, start_time, end_time))
            .flatMap(GenerateSmallRecords.shard_to_files)
            # -> (target_file, (record, start_time, end_time)s)
            .groupByKey()
            # -> (target_file, (record, start_time, end_time)s)
            .mapValues(sorted)
        )
        pprint.PrettyPrinter().pprint(tasks_count.collect())

    @staticmethod
    def shard_to_files(input):
        """(target_dir, (record, header)) -> (task_file, (record, start_time, end_time))"""
        # 1 minute as a record.
        RECORD_DURATION_NS = 60 * (10 ** 9)
        RECORD_FORMAT = '%Y%m%d%H%M00.record'

        target_dir, (record, header) = input
        first_begin_time = (header.begin_time // RECORD_DURATION_NS) * RECORD_DURATION_NS
        last_begin_time = (header.end_time // RECORD_DURATION_NS) * RECORD_DURATION_NS
        for begin_time in range(first_begin_time, last_begin_time + 1, RECORD_DURATION_NS):
            dt = time_utils.msg_time_to_datetime(begin_time)
            target_file = os.path.join(target_dir, dt.strftime(RECORD_FORMAT))
            yield (target_file, (record, begin_time, begin_time + RECORD_DURATION_NS))

if __name__ == '__main__':
    GenerateSmallRecords().run_prod()
