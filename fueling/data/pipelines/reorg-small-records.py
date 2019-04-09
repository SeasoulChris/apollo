#!/usr/bin/env python
import collections
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

from cyber_py.record import RecordReader, RecordWriter

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.time_utils as time_utils


# Config.
RESPECT_TARGET_COMPLETE_MARKER = True
SKIP_EXISTING_TARGET_RECORD = True
# End of configs.


class ReorgSmallRecords(BasePipeline):
    """ReorgSmallRecords pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'reorg-small-records')

    def run_test(self):
        """Run test."""
        sc = self.get_spark_context()
        # RDD(record_path)
        records_rdd = sc.parallelize(['/apollo/docs/demo_guide/demo_3.5.record'])
        # RDD(dir_path)
        whitelist_dirs_rdd = sc.parallelize(['/apollo/docs/demo_guide'])
        # RDD(dir_path)
        blacklist_dirs_rdd = sc.parallelize([])
        origin_prefix = 'docs/demo_guide'
        target_prefix = 'data'
        self.run(records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'modules/data/public-test-small/2019/'
        target_prefix = 'small-records/2019/'

        files = s3_utils.list_files(bucket, origin_prefix).cache()
        records_rdd = files.filter(record_utils.is_record_file)

        whitelist_dirs_rdd = (
            # RDD(COMPLETE_file_path)
            files.filter(lambda path: path.endswith('/COMPLETE'))
            # RDD(task_dir), which has a 'COMPLETE' file inside.
            .map(os.path.dirname))

        blacklist_dirs_rdd = (
            # RDD(file_path), with the target_prefix.
            s3_utils.list_files(bucket, target_prefix)
            # RDD(COMPLETE_file_path)
            .filter(lambda path: path.endswith('/COMPLETE'))
            # RDD(target_dir), which has a 'COMPLETE' file inside.
            .map(os.path.dirname)
            # RDD(task_dir), corresponded to the COMPLETE target_dir.
            .map(lambda path: path.replace(target_prefix, origin_prefix, 1))
        ) if RESPECT_TARGET_COMPLETE_MARKER else self.get_spark_context().emptyRDD()

        summary_receivers = ['xiaoxiangquan@baidu.com']
        self.run(records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
                 origin_prefix, target_prefix, summary_receivers)

    def run(self, records_rdd, whitelist_dirs_rdd, blacklist_dirs_rdd,
            origin_prefix, target_prefix, summary_receivers=None):
        """Run the pipeline with given arguments."""
        input_records = spark_op.log_rdd(
            # PairRDD(task_dir, record), which is in the whitelist and not in the blacklist
            spark_op.filter_keys(records_rdd.keyBy(os.path.dirname),
                                 whitelist_dirs_rdd, blacklist_dirs_rdd)
            # PairRDD(target_dir, record)
            .map(spark_op.do_key(lambda path: path.replace(origin_prefix, target_prefix, 1)))
            # PairRDD(target_dir, record), in absolute style
            .map(lambda (target_dir, record): (s3_utils.abs_path(target_dir),
                                               s3_utils.abs_path(record)))
            # PairRDD(target_dir, (record, header))
            .mapValues(lambda record: (record, record_utils.read_record_header(record)))
            # PairRDD(target_dir, (record, header)), where header is valid
            .filter(lambda (_, (_r, header)): header is not None),
            "InputRecords", glog.info)

        output_records = spark_op.log_rdd(
            # PairRDD(target_dir, (record, header))
            input_records
            # PairRDD(target_file, (record, start_time, end_time))
            .flatMap(self.shard_to_files)
            # PairRDD(target_file, (record, start_time, end_time)s)
            .groupByKey()
            # PairRDD(target_file, (record, start_time, end_time)s)
            .mapValues(sorted),
            "OutputRecords", glog.info)

        finished_tasks = spark_op.log_rdd(
            # PairRDD(target_file, (record, start_time, end_time)s)
            output_records
            # RDD(target_file)
            .map(self.process_file)
            # RDD(target_file)
            .filter(bool)
            # RDD(target_dir)
            .map(os.path.dirname)
            # RDD(target_dir)
            .distinct(),
            "FinishedTasks", glog.info)

        (finished_tasks
            # RDD(target_dir/COMPLETE)
            .map(lambda target_dir: os.path.join(target_dir, 'COMPLETE'))
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

        if SKIP_EXISTING_TARGET_RECORD and os.path.exists(target_file):
            glog.info('Skip generating exist record {}'.format(target_file))
            return target_file
        file_utils.makedirs(os.path.dirname(target_file))
        writer = RecordWriter(0, 0)
        try:
            writer.open(target_file)
        except Exception as e:
            glog.error('Failed to write to target file {}: {}'.format(target_file, e))
            writer.close()
            return None

        known_topics = set()
        for record, start_time, end_time in records:
            glog.info('Read record {}'.format(record))
            try:
                reader = RecordReader(record)
                for msg in reader.read_messages():
                    if msg.timestamp < start_time or msg.timestamp >= end_time:
                        continue
                    if msg.topic not in known_topics:
                        desc = reader.get_protodesc(msg.topic)
                        writer.write_channel(msg.topic, msg.data_type, desc)
                        known_topics.add(msg.topic)
                    writer.write_message(msg.topic, msg.message, msg.timestamp)
            except Exception as err:
                glog.error('Failed to read record {}: {}'.format(record, err))
        writer.close()
        return target_file

    @staticmethod
    def send_summary(task_dirs, receivers):
        """Send summary."""
        if not task_dirs:
            glog.info('No need to send summary for empty result')
            return
        SummaryTuple = collections.namedtuple('Summary', ['TaskDirectory'])
        title = 'Generated small records for {} tasks'.format(len(task_dirs))
        message = [SummaryTuple(TaskDirectory=task_dir) for task_dir in task_dirs]
        try:
            email_utils.send_email_info(title, message, receivers)
        except Exception as error:
            glog.error('Failed to send summary: {}'.format(error))


if __name__ == '__main__':
    ReorgSmallRecords().main()
