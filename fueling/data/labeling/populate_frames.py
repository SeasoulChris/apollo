#!/usr/bin/env python

"""This script extracts sensor messages for labeling"""

from collections import Counter
import glog
import operator
import os
import re
import textwrap
import time

from pyspark.sql import Row
from pyspark.sql import SQLContext

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.data.labeling.populate_utils as populate_utils
import fueling.streaming.streaming_utils as streaming_utils

# Partition records into slices, this is to tell how many records files in one slice
SLICE_SIZE = 4

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
def are_all_channels_available(todo_target, root_dir):
    """Filter record files out if they do not have required channels or simply not readable"""
    _, record_file = todo_target
    if record_file is None or not record_utils.is_record_file(record_file):
        return False
    channels_to_verify = WANTED_CHANNELS.values()
    channels_to_verify.append('/apollo/localization/pose')
    minimal_required_number = 16
    for topic in channels_to_verify:
        lines = list(streaming_utils.load_topic(root_dir, record_file, topic))
        if lines is None or len(lines) < minimal_required_number:
            return False
    return True

def record_to_target_partition(target_record):
    """Shard source record file to a certain partition."""
    target_partition, record_file = target_record
    record_fileparts = os.path.basename(record_file).split('.')
    target_partition += '/{}#SS{}'.format(record_fileparts[0], int(record_fileparts[2])/SLICE_SIZE)
    return (target_partition, record_file)

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

def construct_frames(root_dir, frames):
    """Construct the frame by using given messages.
    Read the according message ONCE again, to avoid playing big messages in memory"""
    target_dir, msgs = frames
    glog.info('Now executors start the hard working.  target_dir: {}'.format(target_dir))

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
    glog.info('Total messages: {}'.format(sum(msg_map.itervalues())))
    
    src_records = list(streaming_utils.target_partition_to_records(root_dir, target_dir, SLICE_SIZE))

    msgs_stream = populate_utils.DataStream(src_records, 
        lambda x: streaming_utils.load_messages(root_dir, x, WANTED_CHANNELS.values()))
    pose_stream = populate_utils.DataStream(src_records, 
        lambda x: streaming_utils.load_messages(root_dir, x, ['/apollo/localization/pose']))
    msgs_iterator = populate_utils.DataStreamIterator(msgs_stream)
    pose_iterator = populate_utils.DataStreamIterator(pose_stream)

    builder_manager = populate_utils. \
        BuilderManager(WANTED_CHANNELS.values(),
                       populate_utils.FramePopulator(root_dir, target_dir, SLICE_SIZE))
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
            msg_new = get_next_message(msg, msg_map, msgs_iterator)
            while msg_new is not None and msg_new.timestamp == msg.timestamp \
                 and msg_new.topic == msg.topic:
                builder_manager.throw_to_pool(populate_utils.MessageStruct(
                    message_struct.message, message_struct.pose_left, message_struct.pose_right))
                msg_new = get_next_message(msg, msg_map, msgs_iterator)
            msg = msg_new
            message_struct = populate_utils.MessageStruct(msg, None, None)

    glog.info('Done with target: {}'.format(target_dir))
    
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

def mark_complete(todo_tasks, target_dir, root_dir):
    """Create COMPLETE file to mark the job done"""
    for task in todo_tasks:
        task_path = os.path.join(root_dir, target_dir)
        task_path = os.path.join(task_path, os.path.basename(task))
        populate_utils.chmod_dir(task_path, 777)
        streaming_utils.write_to_file(\
            os.path.join(task_path, 'COMPLETE'), 'w', '{:.6f}'.format(time.time()))
        populate_utils.chmod_dir(task_path, 755)

class PopulateFramesPipeline(BasePipeline):
    """PopulateFrames pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'populate-frames')

    def run_test(self):
        """Run test."""
        root_dir = '/apollo'
        target_dir = 'modules/data/labeling/generated'
        populate_utils.create_dir_if_not_exist(os.path.join(root_dir, target_dir))
        glog.info('Running TEST, target_dir: {}'.format(os.path.join(root_dir, target_dir)))

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, target_dir)
        glog.info('ToDo tasks: {}'.format(todo_tasks))

        self.run(todo_tasks, root_dir, target_dir)

        glog.info('Task done, marking COMPLETE')
        mark_complete(todo_tasks, target_dir, root_dir)

        glog.info('Labeling: All Done, TEST.')

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        target_dir = 'modules/data/labeling/generated'
        populate_utils.create_dir_if_not_exist(os.path.join(root_dir, target_dir))
        glog.info('Running PROD, target_dir: {}'.format(os.path.join(root_dir, target_dir)))

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, target_dir)
        glog.info('ToDo tasks: {}'.format(todo_tasks))

        self.run(todo_tasks, root_dir, target_dir)

        glog.info('Task done, marking COMPLETE')
        mark_complete(todo_tasks, target_dir, root_dir)

        glog.info('Labeling: All Done, PROD.')

    def run(self, todo_tasks, root_dir, target_dir):
        """Run the pipeline with given arguments."""
        if todo_tasks is None or len(todo_tasks) == 0:
            glog.warn('Labeling: no tasks to process, quit now')
            return

        glog.info('Load messages META data for query')
        spark_context = self.get_spark_context()
        # -> (task_dir)
        msgs_rdd = (spark_context.parallelize(todo_tasks).distinct()
                    # -> (target_dir, task_dir)
                    .keyBy(lambda task_dir: os.path.join(target_dir, os.path.basename(task_dir)))
                    # -> (target_dir, record_files)
                    .flatMapValues(streaming_utils.list_records_for_task)
                    .mapValues(lambda record: record.strip())
                    .filter(lambda record: are_all_channels_available(record, root_dir))
                    # -> (target_partition, record_files)
                    .map(record_to_target_partition)
                    # -> (target_partition, messages_metadata)
                    .flatMapValues(lambda record: streaming_utils \
                        .load_meta_data(root_dir, record, WANTED_CHANNELS.values()))
                    # -> (target_partition, timestamp, topic)
                    .map(lambda (target, meta): (target, meta.timestamp, meta.topic))
                    .cache())

        # Transform RDD to DataFrame, run SQL and tranform back when done
        glog.info('SQL query to search closest sensor messages')
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
         .map(lambda frames: construct_frames(root_dir, frames))
         # -> simply trigger action
         .count())

if __name__ == '__main__':
    PopulateFramesPipeline().run_test()
