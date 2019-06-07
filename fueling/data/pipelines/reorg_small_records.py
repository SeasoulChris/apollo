#!/usr/bin/env python
import collections
import os

import colored_glog as glog
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from cyber_py.record import RecordReader, RecordWriter

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.time_utils as time_utils


# Config.
RESPECT_DEST_COMPLETE_MARKER = True
SKIP_EXISTING_DEST_RECORD = True
MARKER = 'COMPLETE'
# End of configs.


class ReorgSmallRecords(BasePipeline):
    """ReorgSmallRecords pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'reorg-small-records')

    def run_test(self):
        """Run test."""
        # RDD(src_record)
        src_records = self.to_rdd(['/apollo/docs/demo_guide/demo_3.5.record'])
        src_prefix = '/apollo/docs/demo_guide'
        dst_prefix = '/apollo/data'
        self.run(src_records, src_prefix, dst_prefix)

    def run_prod(self):
        """Run prod."""
        src_prefix = 'modules/data/public-test-small/2019/'
        dst_prefix = 'small-records/2019/'

        bos = self.bos()
        # RDD(src_file)
        src_files = self.to_rdd(bos.list_files(src_prefix)).cache()
        # RDD(dst_file)
        dst_files = self.to_rdd(bos.list_files(dst_prefix)).cache()

        is_complete_marker = lambda path: path.endswith(MARKER)
        # RDD(src_dir)
        todo_src_dirs = src_files.filter(is_complete_marker).map(os.path.dirname)

        if RESPECT_DEST_COMPLETE_MARKER:
            # RDD(src_dir), whose dst_dir doesn't have COMPLETE marker.
            todo_src_dirs = todo_src_dirs.subtract(
                # RDD(dst_COMPLETE)
                dst_files.filter(is_complete_marker)
                # RDD(dst_dir)
                .map(os.path.dirname)
                # RDD(src_dir), which has a dst_dir with COMPLETE marker.
                .map(lambda path: path.replace(dst_prefix, src_prefix, 1)))

        # RDD(todo_src_record)
        src_records = spark_op.filter_keys(
            # RDD(src_dir, src_record)
            src_files.filter(record_utils.is_record_file).keyBy(os.path.dirname),
            # RDD(src_dir)
            todo_src_dirs
        ).values()

        self.run(src_records, src_prefix, dst_prefix, email_utils.DATA_TEAM)

    def run(self, src_records, src_prefix, dst_prefix, summary_receivers=None):
        """Run the pipeline with given arguments."""
        partitions = int(os.environ.get('APOLLO_EXECUTORS', 4))
        glog.info('Repartition to: {}'.format(partitions))

        input_records = spark_helper.cache_and_log('InputRecords',
            src_records
            # RDD(src_record)
            .repartition(partitions)
            # PairRDD(src_record, record_header)
            .map(spark_op.value_by(record_utils.read_record_header))
            # PairRDD(src_record, record_header), where header is valid.
            .filter(lambda (_, header): header is not None)
            # PairRDD(dst_dir, (src_record, record_header))
            .keyBy(lambda (record, _): os.path.dirname(record).replace(src_prefix, dst_prefix, 1)))

        output_records = spark_helper.cache_and_log('OutputRecords',
            # PairRDD(target_dir, (record, header))
            input_records
            # PairRDD(target_file, (record, start_time, end_time))
            .flatMap(self.shard_to_files)
            # PairRDD(target_file, (record, start_time, end_time)s)
            .groupByKey()
            # PairRDD(target_file, (record, start_time, end_time)s)
            .mapValues(sorted))

        finished_tasks = spark_helper.cache_and_log('FinishedTasks',
            # PairRDD(target_file, (record, start_time, end_time)s)
            output_records
            # RDD(target_file)
            .map(self.process_file)
            # RDD(target_file)
            .filter(bool)
            # RDD(target_dir)
            .map(os.path.dirname)
            # RDD(target_dir)
            .distinct())

        (finished_tasks
            # RDD(target_dir/COMPLETE)
            .map(lambda target_dir: os.path.join(target_dir, MARKER))
            # Make target_dir/COMPLETE files.
            .foreach(file_utils.touch))

        if summary_receivers:
            self.send_summary(finished_tasks.collect(), summary_receivers)

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

        if SKIP_EXISTING_DEST_RECORD and os.path.exists(target_file):
            glog.info('Skip generating exist record {}'.format(target_file))
            return target_file

        # Read messages and channel information.
        msgs = []
        topic_descs = {}
        for record, start_time, end_time in records:
            glog.info('Read record {}'.format(record))
            try:
                reader = RecordReader(record)
                msgs.extend([msg for msg in reader.read_messages()
                             if start_time <= msg.timestamp < end_time])
                for msg in msgs:
                    if msg.topic not in topic_descs:
                        topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))
                else:
                    glog.error('Failed to read any message from {}'.format(input_record))
                    return target_file
            except Exception as err:
                glog.error('Failed to read record {}: {}'.format(record, err))

        # Check once again to avoid duplicate work after reading.
        if SKIP_EXISTING_DEST_RECORD and os.path.exists(target_file):
            glog.info('Skip generating exist record {}'.format(target_file))
            return target_file
        # Write to record.
        file_utils.makedirs(os.path.dirname(target_file))
        writer = RecordWriter(0, 0)
        try:
            writer.open(target_file)
            for topic, (data_type, desc) in topic_descs.items():
                writer.write_channel(topic, data_type, desc)
            for msg in msgs:
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as e:
            glog.error('Failed to write to target file {}: {}'.format(target_file, e))
            return None
        finally:
            writer.close()
        return target_file

    @staticmethod
    def send_summary(task_dirs, receivers):
        """Send summary."""
        if not task_dirs:
            glog.info('No need to send summary for empty result')
            return
        SummaryTuple = collections.namedtuple('Summary', ['TaskDirectory'])
        title = 'Reorg small records: {}'.format(len(task_dirs))
        message = [SummaryTuple(TaskDirectory=task_dir) for task_dir in task_dirs]
        try:
            email_utils.send_email_info(title, message, receivers)
        except Exception as error:
            glog.error('Failed to send summary: {}'.format(error))


if __name__ == '__main__':
    ReorgSmallRecords().main()
