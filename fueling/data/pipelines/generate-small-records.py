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
        origin_prefix = 'public-test/2019/'
        target_prefix = 'modules/data/public-test-small/2019/'

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
            .map(lambda path: path.replace(target_prefix, origin_prefix, 1)))

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
            # RDD(record)
            .values()
            # PairRDD(target_record, src_record)
            .keyBy(lambda path: path.replace(origin_prefix, target_prefix, 1)),
            "InputRecords", glog.info)

        output_records = spark_op.log_rdd(
            # PairRDD(target_record, src_record)
            input_records
            # PairRDD(src_record, target_record), in absolute style
            .map(lambda (target, source): (s3_utils.abs_path(source), s3_utils.abs_path(target)))
            # RDD(target_file)
            .map(lambda (source, target): self.process_file(source, target))
            # RDD(target_file)
            .filter(bool),
            "OutputRecords", glog.info)

        finished_tasks = spark_op.log_rdd(
            output_records
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
            GenerateSmallRecords.send_summary(finished_tasks.collect(), summary_receivers)


    @staticmethod
    def process_file(input_record, output_record):
        """Process input_record to output_record."""
        glog.info('Processing {} to {}'.format(input_record, output_record))
        if os.path.exists(output_record):
            glog.warn('Skip generating exist record {}'.format(output_record))
            return output_record

        # Read messages and channel information.
        msgs = []
        topic_descs = {}
        try:
            reader = RecordReader(input_record)
            msgs = [msg for msg in reader.read_messages()
                    if msg.topic in GenerateSmallRecords.CHANNELS]
            for msg in msgs:
                if msg.topic not in topic_descs:
                    topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))
        except Exception as err:
            glog.error('Failed to read record {}: {}'.format(record, err))
            return None

        # Write to record.
        file_utils.makedirs(os.path.dirname(output_record))
        writer = RecordWriter(0, 0)
        try:
            writer.open(output_record)
            for topic, (data_type, desc) in topic_descs.items():
                writer.write_channel(topic, data_type, desc)
            for msg in msgs:
                writer.write_message(msg.topic, msg.message, msg.timestamp)
        except Exception as e:
            glog.error('Failed to write to target file {}: {}'.format(output_record, e))
            return None
        finally:
            writer.close()
        return output_record

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
    GenerateSmallRecords().main()
