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
# End of configs.


class GenerateSmallRecords(BasePipeline):
    """GenerateSmallRecords pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'generate-small-records')

    def run_test(self):
        """Run test."""
        # RDD(record_path)
        src_records = self.get_spark_context().parallelize([
            '/apollo/docs/demo_guide/demo_3.5.record'
        ])
        src_prefix = 'docs/demo_guide'
        dst_prefix = 'data'
        self.run(src_records, src_prefix, dst_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        src_prefix = 'public-test/2019/'
        dst_prefix = 'modules/data/public-test-small/2019/'

        # RDD(src_file)
        src_files = s3_utils.list_files(bucket, src_prefix).cache()
        # Only process those COMPLETE folders.
        # RDD(todo_src_record)
        src_records = (
            # PairRDD(src_dir, src_record)
            spark_op.filter_keys(
                # PairRDD(src_dir, src_record)
                src_files.filter(record_utils.is_record_file).keyBy(os.path.dirname),
                # RDD(src_dir), which has COMPLETE marker.
                src_files.filter(lambda path: path.endswith('/COMPLETE').map(os.path.dirname)))
            # RDD(todo_src_record)
            .values()
            .cache())

        spark_op.log_rdd(
            src_records.substract(
                # RDD(dst_file)
                s3_utils.list_files(bucket, dst_prefix)
                # RDD(dst_record)
                .filter(record_utils.is_record_file)
                # RDD(mapped_src_record)
                .map(lambda path: path.replace(dst_prefix, src_prefix, 1))),
            "TodoRecords", glog.info)

        summary_receivers = ['xiaoxiangquan@baidu.com']
        self.run(src_records, src_prefix, dst_prefix, summary_receivers)

    def run(self, src_records, src_prefix, dst_prefix, summary_receivers=None):
        """Run the pipeline with given arguments."""
        output_records = spark_op.log_rdd(
            # RDD(todo_src_record)
            src_records
            # PairRDD(src_record, dst_record)
            .map(spark_op.value_by(lambda path: path.replace(src_prefix, dst_prefix, 1)))
            # PairRDD(src_record, dst_record), in absolute style
            .map(spark_op.do_tuple_elems(s3_utils.abs_path))
            # RDD(dst_record)
            .map(spark_op.do_tuple(self.process_file))
            # RDD(dst_record)
            .filter(bool),
            "OutputRecords", glog.info)

        finished_tasks = spark_op.log_rdd(
            output_records
            # RDD(dst_dir)
            .map(os.path.dirname)
            # RDD(dst_dir)
            .distinct(),
            "FinishedTasks", glog.info)

        (finished_tasks
            # RDD(dst_COMPLETE)
            .map(lambda target_dir: os.path.join(target_dir, 'COMPLETE'))
            # Create dst_COMPLETE files.
            .foreach(file_utils.touch))

        if summary_receivers:
            GenerateSmallRecords.send_summary(finished_tasks.collect(), summary_receivers)


    @staticmethod
    def process_file(input_record, output_record):
        """Process input_record to output_record."""
        glog.info('Processing {} to {}'.format(input_record, output_record))
        if SKIP_EXISTING_DST_RECORDS and os.path.exists(output_record):
            glog.warn('Skip generating exist record {}'.format(output_record))
            return output_record

        # Read messages and channel information.
        msgs = []
        topic_descs = {}
        try:
            reader = RecordReader(input_record)
            msgs = [msg for msg in reader.read_messages() if msg.topic in CHANNELS]
            if not msgs:
                glog.error('Failed to read any message from {}'.format(input_record))
                return None
            for msg in msgs:
                if msg.topic not in topic_descs:
                    topic_descs[msg.topic] = (msg.data_type, reader.get_protodesc(msg.topic))
        except Exception as err:
            glog.error('Failed to read record {}: {}'.format(record, err))
            return None

        # Check once again to avoid duplicate work after reading.
        if SKIP_EXISTING_DST_RECORDS and os.path.exists(output_record):
            glog.warn('Skip generating exist record {}'.format(output_record))
            return output_record
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
