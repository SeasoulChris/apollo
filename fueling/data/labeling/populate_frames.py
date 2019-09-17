#!/usr/bin/env python
"""This script extracts sensor messages for labeling"""

from collections import Counter
from collections import namedtuple
import operator
import os
import textwrap
import time

from absl import flags
from pyspark.sql import Row
from pyspark.sql import SQLContext

from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.data.labeling.populate_utils as populate_utils
import fueling.streaming.streaming_utils as streaming_utils

# Partition records into slices, this is to tell how many records files in one slice
flags.DEFINE_integer('slice_size', 3, 'How many records files in one slice.')
flags.DEFINE_string('labeling_agent', 'scale', 'The labeling company/entity.')
flags.DEFINE_integer('diff', 30, 'Max diff allowed between lidar and camera, in milliseconds.')

# The channels we need to populate
WANTED_CHANNELS = {
    'lidar-128': '/apollo/sensor/lidar128/compensator/PointCloud2',
    'front-6mm': '/apollo/sensor/camera/front_6mm/image/compressed',
    'front-12mm': '/apollo/sensor/camera/front_12mm/image/compressed',
    'rear-6mm': '/apollo/sensor/camera/rear_6mm/image/compressed',
    'left-fisheye': '/apollo/sensor/camera/left_fisheye/image/compressed',
    'right-fisheye': '/apollo/sensor/camera/right_fisheye/image/compressed',
    'front-radar': '/apollo/sensor/radar/front',
    'rear-radar': '/apollo/sensor/radar/rear'
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


def record_to_partition(target_record, slice_size):
    """Shard source record file to a certain partition."""
    target_partition, record_file = target_record
    record_fileparts = os.path.basename(record_file).split('.')
    target_partition += '/{}#SS{}'.format(record_fileparts[0],
                                          int(record_fileparts[2]) / slice_size)
    return (target_partition, record_file)


def create_dataframe(sql_context, msgs_rdd, topics):
    """Create DataFrame for specified topics"""
    return sql_context.createDataFrame(
        msgs_rdd
        .filter(lambda (_1, _2, topic): topic in operator.itemgetter(*topics)(WANTED_CHANNELS))
        .map(lambda x: Row(target=x[0], time=x[1], topic=x[2])))


def get_next_message(msg, msg_map, msgs_iterator):
    """Judiciously decide what the next message is"""
    if msg is not None and \
            msg_map['{}-{}'.format(msg.topic, msg.timestamp)] > 0 and \
            msg.topic != WANTED_CHANNELS['lidar-128']:
        msg_map['{}-{}'.format(msg.topic, msg.timestamp)] -= 1
        return msg
    msg = msgs_iterator.next(lambda x: x.topic in WANTED_CHANNELS.values() and
                             '{}-{}'.format(x.topic, x.timestamp) in msg_map)
    if msg is not None:
        msg_map['{}-{}'.format(msg.topic, msg.timestamp)] -= 1
    return msg


def construct_frames(root_dir, frames, slice_size, agent, diff):
    """Construct the frame by using given messages.
    Read the according message ONCE again, to avoid playing big messages in memory"""
    target_dir, msgs = frames
    logging.info('Now executors start the hard working.  target_dir: {}'.format(target_dir))

    # target_dir looks like:
    #'modules/data/labeling/2019/2019-01-03/2019-01-03-14-56-05/20181113152909#SS0
    # msgs like:
    #(
    #  'lidar_1_topic-lidar_1_time',
    #  'camera_1_topic-camera_1_time',
    #  'lidar_2_topic-lidar_2_time',
    #  'camera_2_topic-camera_2_time'
    #  ...
    #)

    msg_map = Counter(msgs)
    logging.info('Total messages: {}'.format(sum(msg_map.itervalues())))

    src_records = list(streaming_utils.target_partition_to_records(
        root_dir, target_dir, slice_size))

    msgs_stream = populate_utils.DataStream(
        src_records,
        lambda x: streaming_utils.load_messages(root_dir, x, WANTED_CHANNELS.values()))
    pose_stream = populate_utils.DataStream(
        src_records,
        lambda x: streaming_utils.load_messages(root_dir, x, ['/apollo/localization/pose']))
    msgs_iterator = populate_utils.DataStreamIterator(msgs_stream)
    pose_iterator = populate_utils.DataStreamIterator(pose_stream)

    builder_manager = populate_utils.BuilderManager(
        WANTED_CHANNELS.values(),
        populate_utils.FramePopulator(root_dir, target_dir, slice_size))
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
            builder_manager.throw_to_pool(message_struct, agent, diff)
            msg_new = get_next_message(msg, msg_map, msgs_iterator)
            while msg_new is not None and msg_new.timestamp == msg.timestamp \
                    and msg_new.topic == msg.topic:
                builder_manager.throw_to_pool(populate_utils.MessageStruct(message_struct.message,
                                                                           message_struct.pose_left, message_struct.pose_right), agent, diff)
                msg_new = get_next_message(msg, msg_map, msgs_iterator)
            msg = msg_new
            message_struct = populate_utils.MessageStruct(msg, None, None)

    logging.info('Done with target: {}'.format(target_dir))


def get_sql_query(sql_context, msgs_rdd):
    """Get SQL statements for performing the query."""
    lidar_df = create_dataframe(sql_context, msgs_rdd, ('lidar-128',))
    other_sensor_df = create_dataframe(
        sql_context,
        msgs_rdd,
        [x for x in WANTED_CHANNELS if x != 'lidar-128'])
    table_name = 'all_sensor_table'
    lidar_df \
        .join(other_sensor_df
              .withColumnRenamed('time', 'btime')
              .withColumnRenamed('topic', 'btopic'),
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
        """ % {
        'table_name': table_name
    })


def mark_complete(todo_tasks, target_dir, root_dir):
    """Create COMPLETE file to mark the job done"""
    for task in todo_tasks:
        task_path = os.path.join(root_dir, target_dir, os.path.basename(task))
        if not os.path.exists(task_path):
            logging.warning('No data generated for task: {}, check if there are qualified frames'.format(
                task_path))
            continue
        streaming_utils.write_to_file(
            os.path.join(task_path, 'COMPLETE'), 'w', '{:.6f}'.format(time.time()))


class PopulateFramesPipeline(BasePipeline):
    """PopulateFrames pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'populate-frames')

    def run_test(self):
        """Run test."""
        root_dir = '/apollo'
        target_dir = 'modules/data/labeling/generated'
        file_utils.makedirs(os.path.join(root_dir, target_dir))
        logging.info('Running TEST, target_dir: {}'.format(os.path.join(root_dir, target_dir)))

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, target_dir)
        logging.info('ToDo tasks: {}'.format(todo_tasks))

        # TODO: Just show case for email notification, to be updated as something more useful
        notification = namedtuple('Notification', ['todo_tasks', 'root_dir', 'target_dir'])
        message = [notification(task, root_dir, target_dir) for task in todo_tasks]
        email_utils.send_email_info('Frame Population Job Running', message, email_utils.DATA_TEAM)

        self.run(todo_tasks, root_dir, target_dir)

        logging.info('Task done, marking COMPLETE')
        mark_complete(todo_tasks, target_dir, root_dir)

        logging.info('Labeling: All Done, TEST.')

        # TODO: Just show case for email notification, to be updated as something more useful
        email_utils.send_email_info('Frame Population Job Completed', {'Success': 100, 'Fail': 200},
                                    email_utils.DATA_TEAM)

    def run_prod(self):
        """Run prod."""
        bos_client = BosClient()
        root_dir = bos_client.mnt_path
        target_dir = 'modules/data/labeling/generated'
        file_utils.makedirs(bos_client.abs_path(target_dir))
        logging.info('Running PROD, target_dir: {}'.format(bos_client.abs_path(target_dir)))

        _, todo_tasks = streaming_utils.get_todo_records(root_dir, target_dir)
        logging.info('ToDo tasks: {}'.format(todo_tasks))

        self.run(todo_tasks, root_dir, target_dir)

        logging.info('Task done, marking COMPLETE')
        mark_complete(todo_tasks, target_dir, root_dir)

        logging.info('Labeling: All Done, PROD.')

    def run(self, todo_tasks, root_dir, target_dir):
        """Run the pipeline with given arguments."""
        # Creating SQL query will fail and throw if input is empty, so check it here first
        if todo_tasks is None or not todo_tasks:
            logging.warning('Labeling: no tasks to process, quit now')
            return

        logging.info('Load messages META data for query')
        # -> RDD(task_dir), with absolute paths
        msgs_rdd = (self.to_rdd(todo_tasks).distinct()
                    # PairRDD(target_dir, task_dir), target_dir is destination, task_dir is source
                    .keyBy(lambda task_dir: os.path.join(target_dir, os.path.basename(task_dir)))
                    # PairRDD(target_dir, record_files)
                    .flatMapValues(streaming_utils.list_records_for_task)
                    # PairRDD(target_dir, record_files), remove unnecessary characters if any
                    .mapValues(lambda record: record.strip())
                    # PairRDD(target_dir, record_files)
                    .filter(lambda record: are_all_channels_available(record, root_dir))
                    # PairRDD(target_partition, record_files), slice the task into partitions
                    .map(lambda record: record_to_partition(record, self.FLAGS.get('slice_size')))
                    # PairRDD(target_partition, messages_metadata), load messages for each record
                    .flatMapValues(lambda record: streaming_utils
                                   .load_meta_data(root_dir, record, WANTED_CHANNELS.values()))
                    # RDD(target_partition, timestamp, topic)
                    .map(lambda (target, meta): (target, meta.timestamp, meta.topic))
                    .cache())

        # Transform RDD to DataFrame, run SQL and tranform back when done
        logging.info('SQL query to search closest sensor messages')
        sql_context = SQLContext(self.context())
        sql_query = get_sql_query(sql_context, msgs_rdd)
        (sql_context.sql(sql_query)
         # RDD(target, lidar_topic, lidar_time, other_topic, MINIMAL_time)
         .rdd
         .map(list)
         # PairRDD(target, (lidar_topic, lidar_time, other_topic, MINIMAL_time))
         .keyBy(operator.itemgetter(0))
         # PairRDD(target, (topic-time-pair))
         .flatMapValues(lambda x: (('{}-{}'.format(x[1], x[2])), ('{}-{}'.format(x[3], x[4]))))
         # PairRDD(target, (topic-time-pair)s)
         .groupByKey()
         # PairRDD(target, (topic-time-pair)s), process every partition with frames belonging to it
         .map(lambda frames:
              (construct_frames(root_dir, frames, self.FLAGS.get('slice_size'),
                                self.FLAGS.get('labeling_agent'), self.FLAGS.get('diff'))))
         # Simply trigger action
         .count())


if __name__ == '__main__':
    PopulateFramesPipeline().main()
