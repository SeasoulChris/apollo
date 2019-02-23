#!/usr/bin/env python
import datetime
import errno
import os
import pytz

import glog
import pyspark_utils.op as spark_op

from cyber_py.record import RecordReader, RecordWriter

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class GenerateSmallRecords(BasePipeline):
    """GenerateSmallRecords pipeline."""
    RECORD_FORMAT = '%Y%m%d%H%M00.record'
    MSG_TIMEZONE = pytz.timezone('UTC')
    LOCAL_TIMEZONE = pytz.timezone('America/Los_Angeles')

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
            # (task_dir, record)
            spark_op.filter_keys(records_rdd.keyBy(os.path.dirname), whitelist_dirs_rdd)
            # -> (target_dir, record)
            .map(lambda dir_record: (
                s3_utils.abs_path(dir_record[0].replace(origin_prefix, target_prefix, 1)),
                s3_utils.abs_path(dir_record[1])))
            # -> (target_dir, records)
            .groupByKey()
            # -> (target_dir, records), target_dir not exists
            .filter(spark_op.filter_key(
                lambda target_dir: not os.path.exists(os.path.join(target_dir, 'COMPLETE'))))
            # -> (target_dir, sorted_records)
            .mapValues(sorted)
            # -> target_dir
            .map(GenerateSmallRecords.generate_small_records)
            # -> target_dir, which is finished
            .filter(lambda target_dir: target_dir is not None)
            # -> target_dir/COMPLETE
            .map(lambda target_dir: os.path.join(target_dir, 'COMPLETE'))
            # Touch file.
            .map(os.mknod)
            # Trigger actions.
            .count())
        glog.info('Finished %d tasks!' % tasks_count)

    @staticmethod
    def generate_small_records(input):
        """(target_dir, records) -> target_dir"""
        target_dir, records = input
        glog.info('Processing {} records to directory {}'.format(len(records), target_dir))
        # Create folder.
        try:
            os.makedirs(target_dir)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise

        writer = RecordWriter(0, 0)
        current_output_file = None
        known_channels = set()
        output_messages = 0
        for record in records:
            glog.info('Processing {}'.format(record))
            try:
                reader = RecordReader(record)
                for msg in reader.read_messages():
                    if msg.topic not in GenerateSmallRecords.CHANNELS:
                        continue
                    dt = datetime.datetime.fromtimestamp(
                            msg.timestamp / (10 ** 9), GenerateSmallRecords.MSG_TIMEZONE
                        ).astimezone(GenerateSmallRecords.LOCAL_TIMEZONE)
                    output_file = os.path.join(target_dir,
                                               dt.strftime(GenerateSmallRecords.RECORD_FORMAT))
                    if output_file != current_output_file:
                        if current_output_file is not None:
                            writer.close()
                        current_output_file = output_file
                        writer.open(current_output_file)
                    if msg.topic not in known_channels:
                        writer.write_channel(msg.topic, msg.data_type,
                                             reader.get_protodesc(msg.topic))
                        known_channels.add(msg.topic)
                    writer.write_message(msg.topic, msg.message, msg.timestamp)
                    output_messages += 1
            except Exception:
                glog.error('Failed to read record {}'.format(record))
        if current_output_file is not None:
            writer.close()
        return target_dir if output_messages > 0 else None


if __name__ == '__main__':
    GenerateSmallRecords().run_prod()
