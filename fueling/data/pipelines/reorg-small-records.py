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


class TaskProcessor(object):
    """Process a task."""
    TAGRTE_RECORD_FORMAT = '%Y%m%d%H%M00.record'

    def __init__(self, records, target_dir):
        self.records = records
        self.target_dir = target_dir
        glog.info('Processing {} records to {}'.format(len(records), target_dir))
        self.writer = None
        self.current_target_file = None
        self.current_known_topics = set()

    def _reset_writer(self, output_record=None):
        try:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            if output_record is not None:
                self.writer = RecordWriter(0, 0)
                self.writer.open(output_record)
        except Exception as error:
            glog.error('Failed to reset writer to {}: {}'.format(output_record, error))
            self.writer.close()
            self.writer = None
            self.current_target_file = None
            return False
        self.current_target_file = output_record
        self.current_known_topics = set()
        return True

    def process_task(self):
        """Process 1 task."""
        file_utils.makedirs(self.target_dir)
        processed_records = 0
        for record in self.records:
            if self.process_record(record):
                processed_records += 1
        self._reset_writer()
        return target_dir if processed_records > 0 else None

    def process_record(self, record):
        """Process 1 record."""
        glog.info('Read record {}'.format(record))
        try:
            reader = RecordReader(record)
            for msg in reader.read_messages():
                dt = time_utils.msg_time_to_datetime(msg.timestamp)
                target_file = os.path.join(self.target_dir, dt.strftime(self.TAGRTE_RECORD_FORMAT))
                if self.current_target_file != target_file:
                    if not self._reset_writer(target_file):
                        return False
                if msg.topic not in self.current_known_topics:
                    desc = reader.get_protodesc(msg.topic)
                    if desc:
                        self.writer.write_channel(msg.topic, msg.data_type, desc)
                        self.current_known_topics.add(msg.topic)
                self.writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as error:
            glog.error('Failed to read record {}: {}'.format(record, error))
            return False
        return True


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
        origin_prefix = 'modules/data/public-test-small/2018/'
        target_prefix = 'small-records/2018/'

        files = s3_utils.list_files(bucket, origin_prefix).cache()
        records_rdd = files.filter(record_utils.is_record_file)

        whitelist_dirs_rdd = (
            # RDD(COMPLETE_file_path)
            files.filter(lambda path: path.endswith('/COMPLETE'))
            # RDD(task_dir), which has a 'COMPLETE' file inside.
            .map(os.path.dirname))

        blacklist_dirs_rdd = self.get_spark_context().parallelize([])
        """
        blacklist_dirs_rdd = (
            # RDD(file_path), with the target_prefix.
            s3_utils.list_files(bucket, target_prefix)
            # RDD(COMPLETE_file_path)
            .filter(lambda path: path.endswith('/COMPLETE'))
            # RDD(target_dir), which has a 'COMPLETE' file inside.
            .map(os.path.dirname)
            # RDD(task_dir), corresponded to the COMPLETE target_dir.
            .map(lambda path: path.replace(target_prefix, origin_prefix, 1)))
        """

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
            .map(spark_op.do_key(lambda path: path.replace(origin_prefix, target_prefix, 1))),
            # PairRDD(target_dir, record), in absolute style
            .map(lambda (target_dir, record): (s3_utils.abs_path(target_dir),
                                               s3_utils.abs_path(record))),
            "InputRecords", glog.info)

        output_dirs = spark_op.log_rdd(
            # PairRDD(target_dir, record)
            input_records
            # PairRDD(target_dir, records)
            .groupByKey()
            # PairRDD(target_dir, records)
            .mapValues(sorted),
            "OutputDirs", glog.info)

        finished_tasks = spark_op.log_rdd(
            # PairRDD(target_dir, records)
            output_dirs
            # RDD(target_dir), or None if the task is a failure.
            .map(lambda (target_dir, records): TaskProcessor(records, target_dir).process_task())
            # RDD(target_dir), which is valid.
            .filter(bool),
            "FinishedTasks", glog.info)

        (finished_tasks
            # RDD(target_dir/COMPLETE)
            .map(lambda target_dir: os.path.join(target_dir, 'COMPLETE'))
            # Make target_dir/COMPLETE files.
            .foreach(file_utils.touch))

        if summary_receivers:
            GenerateSmallRecords.send_summary(finished_tasks.collect(), summary_receivers)

    @staticmethod
    def send_summary(task_dirs, receivers):
        """Send summary."""
        if len(task_dirs) == 0:
            glog.info('No need to send summary for empty result')
            return
        SummaryTuple = collections.namedtuple('Summary', ['TaskDirectory'])
        title = 'Generated small records for {} tasks'.format(len(task_dirs))
        message = [SummaryTuple(TaskDirectory=task_dir) for task_dir in task_dirs]
        email_utils.send_email_info(title, message, receivers)


if __name__ == '__main__':
    GenerateSmallRecords().main()
