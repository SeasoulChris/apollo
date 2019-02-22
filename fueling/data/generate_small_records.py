#!/usr/bin/env python
from datetime import datetime
import errno
import os

import glog
import pyspark_utils.op as spark_op

from cyber_py.record import RecordReader, RecordWriter

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class GenerateSmallRecords(BasePipeline):
    """GenerateSmallRecords pipeline."""
    PARTITION = 100  # This should be related to worker count.
    FILE_SEGMENT_INTERVAL_SEC = 60
    RECORD_FORMAT = '%Y-%m-%d_%H-%M-%S.record'
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
        origin_prefix = '/apollo/docs/demo_guide'
        target_prefix = '/apollo/data'
        self.run(records_rdd, whitelist_dirs_rdd, origin_prefix, target_prefix)
    
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
        self.run(records_rdd, whitelist_dirs_rdd, origin_prefix, target_prefix)

    def run(self, records_rdd, whitelist_dirs_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        tasks_count = (
            # -> (task_dir, record)
            spark_op.filter_keys(records_rdd.keyBy(os.path.dirname), whitelist_dirs_rdd)
            # -> (target_dir, record)
            .map(lambda dir_record: (
                s3_utils.abs_path(dir_record[0].replace(origin_prefix, target_prefix, 1)),
                s3_utils.abs_path(dir_record[1])))
            # -> (target_dir, record), target_dir/COMPLETE not exists
            .filter(spark_op.filter_key(
                lambda target_dir: not os.path.exists(os.path.join(target_dir, 'COMPLETE'))))
            .repartition(GenerateSmallRecords.PARTITION)
            # -> (target_dir, msg)
            .flatMapValues(record_utils.read_record(GenerateSmallRecords.CHANNELS))
            # -> (target_file, msg)
            .map(lambda dir_msg: (
                os.path.join(dir_msg[0], datetime.fromtimestamp(
                    dir_msg[1].timestamp / (10 ** 9)).strftime('%Y%m%d%H%M00.record')),
                dir_msg[1]))
            # -> (target_file, msgs)
            .groupByKey()
            # -> (target_file, sorted_msgs)
            .mapValues(lambda msgs: sorted(msgs, key=lambda msg: msg.timestamp))
            # -> (target_file)
            .map(record_utils.write_record)
            # -> (target_dir)
            .map(os.path.dirname)
            # -> (target_dir)
            .distinct()
            # -> (target_dir/COMPLETE)
            .map(lambda path: os.path.join(path, 'COMPLETE'))
            # Touch file.
            .map(os.mknod)
            # Trigger actions.
            .count())
        glog.info('Finished %d tasks!' % tasks_count)


if __name__ == '__main__':
    GenerateSmallRecords().run_prod()
