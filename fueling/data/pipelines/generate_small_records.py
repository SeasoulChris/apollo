#!/usr/bin/env python
"""General small records"""

import collections
import os

from absl import flags
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from cyber_py.record import RecordReader, RecordWriter

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


PROCESS_LAST_N_DAYS = 30


# Config.
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
SKIP_EXISTING_DST_RECORDS = True
MARKER = 'COMPLETE'
# End of configs.


class GenerateSmallRecords(BasePipeline):
    """GenerateSmallRecords pipeline."""

    def run_test(self):
        """Run test."""
        # RDD(record_path)
        todo_records = self.to_rdd(['/apollo/modules/data/fuel/testdata/data/small.record'])
        src_prefix = '/apollo/modules/data'
        dst_prefix = '/tmp/generate_small_records'
        self.run(todo_records, src_prefix, dst_prefix)

    def run_prod(self):
        """Run prod."""
        src_prefix = 'public-test/2019/'
        dst_prefix = 'modules/data/public-test-small/2019/'

        bos = self.bos()
        # RDD(src_file)
        src_files = self.to_rdd(bos.list_files(src_prefix)).cache()
        dst_files = self.to_rdd(bos.list_files(dst_prefix)).cache()
        # Only process those COMPLETE folders.

        def is_marker(path): return path.endswith(MARKER)
        # PairRDD(src_dir, src_record)
        src_dir_and_records = spark_op.filter_keys(
            # PairRDD(src_dir, src_record)
            src_files.filter(record_utils.is_record_file).keyBy(os.path.dirname),
            # RDD(src_dir), which has COMPLETE marker.
            src_files.filter(is_marker).map(os.path.dirname)
        ).cache()

        # RDD(todo_record)
        todo_records = (
            # PairRDD(src_dir, src_record)
            src_dir_and_records
            # RDD(todo_record)
            .values()
            # RDD(todo_record), which is like /mnt/bos/small-records/2019/2019-09-09/...
            .filter(record_utils.filter_last_n_days_records(PROCESS_LAST_N_DAYS)))

        if SKIP_EXISTING_DST_RECORDS:
            todo_records = todo_records.subtract(
                # RDD(dst_record)
                dst_files.filter(record_utils.is_record_file)
                # RDD(mapped_src_record)
                .map(lambda path: path.replace(dst_prefix, src_prefix, 1)))

            partitions = int(os.environ.get('APOLLO_EXECUTORS', 4))
            logging.info('Repartition to: {}'.format(partitions))
            todo_records = todo_records.repartition(partitions).cache()

            # Mark dst_dirs which have finished.
            spark_helper.cache_and_log('SupplementMarkers',
                                       # RDD(src_dir)
                                       src_dir_and_records.keys()
                                       # RDD(src_dir), which is unique.
                                       .distinct()
                                       # RDD(src_dir), which in src_dirs but not in todo_dirs.
                                       .subtract(todo_records.map(os.path.dirname).distinct())
                                       # RDD(dst_dir)
                                       .map(lambda path: path.replace(src_prefix, dst_prefix, 1))
                                       # RDD(dst_MARKER)
                                       .map(lambda path: os.path.join(path, MARKER))
                                       # RDD(dst_MARKER), which doesn't exist.
                                       .subtract(dst_files.filter(is_marker))
                                       # RDD(dst_MARKER), which is touched.
                                       .map(file_utils.touch))

        spark_helper.cache_and_log('TodoRecords', todo_records)
        self.run(todo_records, src_prefix, dst_prefix, email_utils.DATA_TEAM)

    def run(self, todo_records, src_prefix, dst_prefix, summary_receivers=None):
        """Run the pipeline with given arguments."""
        output_records = spark_helper.cache_and_log('OutputRecords',
                                                    # RDD(todo_src_record)
                                                    todo_records
                                                    # PairRDD(src_record, dst_record)
                                                    .map(spark_op.value_by(lambda path: path.replace(src_prefix, dst_prefix, 1)))
                                                    # RDD(dst_record)
                                                    .map(spark_op.do_tuple(self.process_file))
                                                    # RDD(dst_record)
                                                    .filter(bool))

        # RDD(dst_dir)
        finished_dirs = spark_helper.cache_and_log('FinishedDirs',
                                                   output_records.map(os.path.dirname).distinct())
        # Touch dst_dir/COMPLETE
        finished_dirs.foreach(lambda dst_dir: file_utils.touch(os.path.join(dst_dir, MARKER)))
        if summary_receivers:
            GenerateSmallRecords.send_summary(finished_dirs.collect(), summary_receivers)

    @staticmethod
    def process_file(src_record, dst_record):
        """Process src_record to dst_record."""
        logging.info('Processing {} to {}'.format(src_record, dst_record))
        if SKIP_EXISTING_DST_RECORDS and os.path.exists(dst_record):
            logging.warning('Skip generating exist record {}'.format(dst_record))
            return dst_record

        # Read messages and channel information.
        msgs = []
        topic_descs = {}
        try:
            reader = RecordReader(src_record)
            msgs = [msg for msg in reader.read_messages() if msg.topic in CHANNELS]
            if len(msgs) == 0:
                logging.error('Failed to read any message from {}'.format(src_record))
                return dst_record

            for msg in msgs:
                if msg.topic not in topic_descs:
                    topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))
        except Exception as err:
            logging.error('Failed to read record {}: {}'.format(src_record, err))
            return None

        # Check once again to avoid duplicate work after reading.
        if SKIP_EXISTING_DST_RECORDS and os.path.exists(dst_record):
            logging.warning('Skip generating exist record {}'.format(dst_record))
            return dst_record
        # Write to record.
        file_utils.makedirs(os.path.dirname(dst_record))
        writer = RecordWriter(0, 0)
        try:
            writer.open(dst_record)
            for topic, (data_type, desc) in topic_descs.items():
                writer.write_channel(topic, data_type, desc)
            for msg in msgs:
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as e:
            logging.error('Failed to write to target file {}: {}'.format(dst_record, e))
            return None
        finally:
            writer.close()
        return dst_record

    @staticmethod
    def send_summary(task_dirs, receivers):
        """Send summary."""
        if not task_dirs:
            logging.info('No need to send summary for empty result')
            return
        SummaryTuple = collections.namedtuple('Summary', ['TaskDirectory'])
        title = 'Generated small records: {}'.format(len(task_dirs))
        message = [SummaryTuple(TaskDirectory=task_dir) for task_dir in task_dirs]
        try:
            email_utils.send_email_info(title, message, receivers)
        except Exception as error:
            logging.error('Failed to send summary: {}'.format(error))


if __name__ == '__main__':
    GenerateSmallRecords().main()
