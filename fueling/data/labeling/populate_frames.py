#!/usr/bin/env python

"""This script extracts sensor messages for labeling"""

from collections import Counter
import operator
import os
import re
import textwrap

from pyspark.sql import Row
from pyspark.sql import SQLContext

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.data.labeling.populate_utils as populate_utils

# Partition records into slices, this is to tell how many records files in one slice
SLICE_SIZE = 3

# The channels we need to populate
WANTED_CHANNELS = {
    'lidar-128':'/apollo/sensor/lidar128/compensator/PointCloud2',
    'front-6mm':'/apollo/sensor/camera/front_6mm/image/compressed',
    'front-12mm':'/apollo/sensor/camera/front_12mm/image/compressed',
    'rear-6mm':'/apollo/sensor/camera/rear_6mm/image/compressed',
    'left-fisheye':'/apollo/sensor/camera/left_fisheye/image/compressed',
    'right-fisheye':'/apollo/sensor/camera/right_fisheye/image/compressed',
    'front-radar':'/apollo/sensor/radar/front',
    'rear-radar':'/apollo/sensor/radar/rear'
}

# Helper functions
def get_todo_files(bucket, original_prefix, target_prefix):
    """Get to be processed files in rdd format."""
    files = s3_utils.list_files(bucket, original_prefix).persist()

    # (task_dir, _), which is "public-test/..." with 'COMPLETE' mark.
    complete_dirs = (files
                     .filter(lambda path: path.endswith('/COMPLETE'))
                     .keyBy(os.path.dirname))

    # target_dir
    processed_dirs = s3_utils.list_dirs(bucket, target_prefix).keyBy(lambda path: path)

    # Find all todo jobs.
    todo_files = (files
                  .filter(record_utils.is_record_file)  # -> record
                  .keyBy(os.path.dirname)             # -> (task_dir, record)
                  .join(complete_dirs)                # -> (task_dir, (record, _))
                  .mapValues(operator.itemgetter(0))  # -> (task_dir, record)
                  .map(spark_op.do_key(
                      lambda src_dir: src_dir.replace(original_prefix, target_prefix, 1)))
                  .subtractByKey(processed_dirs)      # -> (target_dir, record) unprocessed files
                  .cache())

    return todo_files

def record_to_target_partition(target_record):
    """Shard source record file to a certain partition."""
    target_partition, record_file = target_record
    record_fileparts = os.path.basename(record_file).split('.')
    target_partition += '/{}#SS{}'.format(record_fileparts[0], int(record_fileparts[2])/SLICE_SIZE)
    return (target_partition, record_file)

def target_partition_to_records(root_dir, original_prefix, target_prefix, target_partition):
    """Revert a partition to original record files"""
    seq = int(re.search(r'[\S]*\d{14}#SS([\d]*)$', target_partition, re.M|re.I).group(1))
    src_record_file = target_partition.replace(target_prefix, original_prefix, 1).rsplit('#', 1)[0]
    for rec in range(seq * SLICE_SIZE, (seq+1) * SLICE_SIZE):
        ret_record = '{}.record.{:05d}'.format(src_record_file, rec)
        if os.path.exists(os.path.join(root_dir, ret_record)):
            yield ret_record

def create_dataframe(sql_context, msgs_rdd, topics):
    """Create DataFrame for specified topics"""
    return sql_context.createDataFrame(
        msgs_rdd \
        .filter(lambda (_1, _2, topic): topic in operator.itemgetter(*topics)(WANTED_CHANNELS)) \
        .map(lambda x: Row(target=x[0], time=x[1], topic=x[2])))

def get_next_message(msg, msg_map, msgs_iterator):
    """Judiciously decide what the next message is"""
    if msg is not None and \
            msg_map['{}-{}'.format(msg.topic, msg.timestamp)] > 0 and \
            msg.topic != WANTED_CHANNELS['lidar-128']:
        msg_map['{}-{}'.format(msg.topic, msg.timestamp)] -= 1
        return msg
    msg = msgs_iterator.next(lambda x: x.topic in WANTED_CHANNELS.values() and \
                            '{}-{}'.format(x.topic, x.timestamp) in msg_map)
    if msg is not None:
        msg_map['{}-{}'.format(msg.topic, msg.timestamp)] -= 1
    return msg

def construct_frames(root_dir, original_prefix, target_prefix, frames):
    """Construct the frame by using given messages.
    Read the according message ONCE again, to avoid playing big messages in memory"""
    target_dir, msgs = frames

    #target_dir looks like:
    #'modules/data/labeling/2019/2019-01-03/2019-01-03-14-56-05/20181113152909#SS0
    #msgs like:
    #(
    #  'lidar_1_topic-lidar_1_time',
    #  'camera_1_topic-camera_1_time',
    #  'lidar_2_topic-lidar_2_time',
    #  'camera_2_topic-camera_2_time'
    #  ...
    #)
    msg_map = Counter(msgs)
    msgs_stream = populate_utils.DataStream(
        [os.path.join(root_dir, x) for x in \
            target_partition_to_records(root_dir, original_prefix, target_prefix, target_dir)],
        populate_utils.read_messages_func)
    msgs_iterator = populate_utils.DataStreamIterator(msgs_stream)
    pose_iterator = populate_utils.DataStreamIterator(msgs_stream)

    builder_manager = populate_utils. \
        BuilderManager(WANTED_CHANNELS.values(),
                       populate_utils.FramePopulator(os.path.join(root_dir, target_dir)))
    msg = get_next_message(None, msg_map, msgs_iterator)
    message_struct = populate_utils.MessageStruct(msg, None, None)
    pose = pose_iterator.next(lambda x: x.topic == '/apollo/localization/pose')

    while msg is not None and pose is not None:
        if pose.timestamp < msg.timestamp:
            if message_struct.pose_left is None or message_struct.pose_left < pose.timestamp:
                message_struct.pose_left = pose
            pose = pose_iterator.next(lambda x: x.topic == '/apollo/localization/pose')
        else:
            if message_struct.pose_right is None or message_struct.pose_right > pose.timestamp:
                message_struct.pose_right = pose
            builder_manager.throw_to_pool(message_struct)
            msg = get_next_message(msg, msg_map, msgs_iterator)
            message_struct = populate_utils.MessageStruct(msg, None, None)

def get_sql_query(sql_context, msgs_rdd):
    """Get SQL statements for performing the query."""
    lidar_df = create_dataframe(sql_context, msgs_rdd, ('lidar-128',))
    other_sensor_df = create_dataframe(
        sql_context,
        msgs_rdd,
        [x for x in WANTED_CHANNELS if x != 'lidar-128'])

    table_name = 'all_sensor_table'
    lidar_df \
        .join(other_sensor_df \
                .withColumnRenamed('time', 'btime') \
                .withColumnRenamed('topic', 'btopic'), \
            lidar_df.target == other_sensor_df.target) \
        .drop(other_sensor_df.target) \
        .registerTempTable(table_name)

    return textwrap.dedent("""
        SELECT A.target, A.topic, A.time, A.btopic, A.btime FROM %(table_name)s A
        INNER JOIN
        (
            SELECT target, time, topic, btopic, MIN(ABS(time-btime)) as mindiff
            FROM %(table_name)s 
            GROUP BY target, time, topic, btopic
        ) B on A.target=B.target AND A.time=B.time AND A.topic=B.topic
        WHERE ABS(A.time-A.btime)=B.mindiff
        """%{
            'table_name': table_name
            })

def mark_complete(todo_files, root_dir):
    """Create COMPLETE file to mark the job done"""
    todo_files \
        .keys() \
        .distinct() \
        .map(lambda path: os.path.join(root_dir, os.path.join(path, 'COMPLETE'))) \
        .map(os.mknod) \
        .count()

class PopulateFramesPipeline(BasePipeline):
    """PopulateFrames pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'populate-frames')

    def run_test(self):
        """Run test."""
        spark_context = self.get_spark_context()
        original_prefix = 'modules/data/fuel/testdata/data/labeling/original/'
        target_prefix = 'modules/data/fuel/testdata/data/labeling/generated/'
        root_dir = '/apollo'
        # This is for testing, and files locations are subject to change
        todo_files = spark_context.parallelize([
            ('modules/data/fuel/testdata/data/labeling/generated',
             'modules/data/fuel/testdata/data/labeling/original/20190220215801.record.00000'),
            ('modules/data/fuel/testdata/data/labeling/generated',
             'modules/data/fuel/testdata/data/labeling/original/20190220215801.record.00002'),
            ('modules/data/fuel/testdata/data/labeling/generated',
             'modules/data/fuel/testdata/data/labeling/original/20190220215801.record.00003')
        ])
        self.run(todo_files, root_dir, original_prefix, target_prefix)
        mark_complete(todo_files, root_dir)
        print 'All Done: labeling, TEST.'

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        # Original records are public-test/path/to/*.record, sharded to M
        original_prefix = 'public-test/2019/'
        # Target labeling frames folder containing the populated N frames
        target_prefix = 'modules/data/labeling/2019/'
        root_dir = s3_utils.S3_MOUNT_PATH
        todo_files = get_todo_files(bucket, original_prefix, target_prefix)
        self.run(todo_files, root_dir, original_prefix, target_prefix)
        mark_complete(todo_files, root_dir)
        print 'All Done: labeling, PROD.'

    def run(self, todo_files, root_dir, original_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        msgs_rdd = (todo_files     # -> (target_dir, record_file)
                    # -> (target_dir_partition, record_file)
                    .map(record_to_target_partition)
                    # -> (target_dir_partition, message)
                    .flatMapValues(record_utils.read_record(WANTED_CHANNELS.values()))
                    # -> (target_dir_partition, timestamp, topic)
                    .map(lambda (target, msg): (target, msg.timestamp, msg.topic))
                    .cache())

        # Transform RDD to DataFrame, run SQL and tranform back when done
        spark_context = self.get_spark_context()
        sql_context = SQLContext(spark_context)
        sql_query = get_sql_query(sql_context, msgs_rdd)
        (sql_context
         .sql(sql_query)
         # -> (target, lidar_topic, lidar_time, other_topic, MINIMAL_time)
         .rdd
         .map(list)
         # -> (target, (lidar_topic, lidar_time, other_topic, MINIMAL_time))
         .keyBy(operator.itemgetter(0))
         # -> (target, (topic-time-pair))
         .flatMapValues(lambda x: (('{}-{}'.format(x[1], x[2])), ('{}-{}'.format(x[3], x[4]))))
         # -> (target, (topic-time-pair))
         #.distinct()
         # -> <target, {(topic-time-pair)}>
         .groupByKey()
         # -> process every partition, with all frames belonging to it
         .map(lambda x: construct_frames(root_dir, original_prefix, target_prefix, x))
         # -> simply trigger action
         .count())

if __name__ == '__main__':
    PopulateFramesPipeline().run_test()
