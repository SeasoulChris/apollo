#!/usr/bin/env python
import errno
import os

import pyspark_utils.op as spark_op

from cyber_py.record import RecordWriter

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
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
        root_dir = '/apollo'
        records_rdd = sc.parallelize(['docs/demo_guide/demo_3.5.record'])
        whitelist_dirs_rdd = sc.parallelize(['docs/demo_guide'])
        blacklist_dirs_rdd = sc.parallelize([])
        origin_prefix = 'docs/demo_guide'
        target_prefix = 'data'
        self.run(root_dir, records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
                 origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        bucket = 'apollo-platform'
        # Original records are public-test/path/to/*.record, sharded to M.
        origin_prefix = 'public-test/2019/2019-01'
        # We will process them to small-records/path/to/*.record, sharded to N.
        target_prefix = 'small-records/2019/2019-01'

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
        self.run(root_dir, records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
                 origin_prefix, target_prefix)

    def run(self, root_dir, records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
            origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        partitions = int(os.environ.get('APOLLO_EXECUTORS', 20)) * 10
        glog.info('Run pipeline in {} partitions'.format(partitions))

        # (task_dir, record)
        todo_jobs = spark_op.filter_keys(records_rdd.keyBy(os.path.dirname), whitelist_dirs_rdd)
        tasks_count = (
            # -> (task_dir, record)
            spark_op.substract_keys(todo_jobs, blacklist_dirs_rdd)
            # -> (target_dir, record)
            .map(spark_op.do_key(lambda path: path.replace(origin_prefix, target_prefix, 1)))
            # -> (target_dir, record), in absolute path style.
            .map(lambda dir_record: (os.path.join(root_dir, dir_record[0]),
                                     os.path.join(root_dir, dir_record[1])))
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
            .persist()
            # -> target_file
            .map(GenerateSmallRecords.process_file)
            # -> target_file
            .filter(lambda path: path is not None)
            # -> target_dir
            .map(os.path.dirname)
            # -> target_dir
            .distinct()
            # -> target_dir/COMPLETE
            .map(lambda target_dir: os.path.join(target_dir, 'COMPLETE'))
            # Touch file.
            .map(GenerateSmallRecords.touch_file)
            # Trigger actions.
            .count())
        glog.info('Processed {} tasks'.format(tasks_count))

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

    @staticmethod
    def process_file(input):
        """(target_file, (record, start_time, end_time)s) -> target_file"""
        target_file, records = input
        glog.info('Processing {} records to {}'.format(len(records), target_file))

        if os.path.exists(target_file) and record_utils.read_record_header(target_file) is not None:
            glog.info('Skip generating exist record {}'.format(target_file))
            return target_file

        try:
            os.makedirs(os.path.dirname(target_file))
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise

        _, start_time, end_time = records[0]
        reader = record_utils.read_record(GenerateSmallRecords.CHANNELS, start_time, end_time)
        writer = RecordWriter(0, 0)
        try:
            writer.open(target_file)
        except Exception as e:
            glog.error('Failed to write to target file {}: {}'.format(target_file, e))
            writer.close()
            return None

        topics = set()
        for record, _, _ in records:
            for msg in reader(record):
                if msg.topic not in topics:
                    # As a generated record, we ignored the proto desc.
                    writer.write_channel(msg.topic, msg.data_type, '')
                    topics.add(msg.topic)
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        writer.close()
        return target_file

    @staticmethod
    def touch_file(path):
        """Touch file."""
        if not os.path.exists(path):
            glog.info('Touch file {}'.format(path))
            os.mknod(path)
        return path


if __name__ == '__main__':
    GenerateSmallRecords().run_prod()
