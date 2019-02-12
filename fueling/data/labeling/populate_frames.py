#!/usr/bin/env python

import datetime
import operator
import os
import pprint
import re
import subprocess
import sys
import textwrap
import time
from collections import defaultdict

from cyber_py import record
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark.sql import functions as F

import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.spark_utils as spark_utils
from fueling.data.labeling.populate_utils import FramePopulator

kBucket = 'apollo-platform'
# Original records are public-test/path/to/*.record, sharded to M.
kOriginPrefix = 'public-test/2019/'
# Target labeling frames folder, which contains the populated N frames corresponding to original records
kTargetPrefix = 'modules/data/labeling/2019/'
# Partition records into slices, this is to tell how many records files in one slice
kSliceSize = 3

kWantedChannels = {
  'lidar-128': '/apollo/sensor/lidar128/compensator/PointCloud2', 
  'front-6mm': '/apollo/sensor/camera/front_6mm/image/compressed',
  'front-12mm': '/apollo/sensor/camera/front_12mm/image/compressed',
  'rear-6mm': '/apollo/sensor/camera/rear_6mm/image/compressed',
  'left-fisheye': '/apollo/sensor/camera/left_fisheye/image/compressed',
  'right-fisheye': '/apollo/sensor/camera/right_fisheye/image/compressed',
  'front-radar': '/apollo/sensor/radar/front',
  'rear-radar': '/apollo/sensor/radar/rear',
  'pose': '/apollo/localization/pose'
}

def get_todo_files():
    """Get to be processed files in rdd format."""
    files = s3_utils.ListFiles(kBucket, kOriginPrefix).persist()

    # (task_dir, _), which is "public-test/..." with 'COMPLETE' mark.
    complete_dirs = (files
        .filter(lambda path: path.endswith('/COMPLETE'))
        .keyBy(os.path.dirname))

    # target_dir
    processed_dirs = s3_utils.ListDirs(kBucket, kTargetPrefix).keyBy(lambda path: path)

    # Find all todo jobs.
    todo_files = (files
        .filter(record_utils.IsRecordFile)  # -> record
        .keyBy(os.path.dirname)             # -> (task_dir, record)
        .join(complete_dirs)                # -> (task_dir, (record, _))
        .mapValues(operator.itemgetter(0))  # -> (task_dir, record)
        .map(spark_utils.MapKey(lambda src_dir: src_dir.replace(kOriginPrefix, kTargetPrefix, 1)))
                                            # -> (target_dir, record)
        .subtractByKey(processed_dirs)      # -> (target_dir, record), which is not processed
        .cache())
    
    return todo_files

def record_to_target_partition(target_record):
    """Shard source record file to a certain partition."""
    target_partition, record = target_record
    record_fileparts = os.path.basename(record).split('.')
    target_partition += '/{}#SS{}'.format(record_fileparts[0], int(record_fileparts[2])/kSliceSize)
    return (target_partition, record)

def target_partition_to_records(target_partition):
    """Revert a partition to original record files"""
    seq = int(re.search(r'[\S]*\d{14}#SS([\d]*)$', target_partition, re.M|re.I).group(1))
    src_record_file = target_partition.replace(kTargetPrefix, kOriginPrefix, 1).rsplit('#', 1)[0]
    for rec in range(seq * kSliceSize, (seq+1) * kSliceSize):
        ret_record = '{}.record.{:05d}'.format(src_record_file, rec)
        if os.path.exists(os.path.join(s3_utils.S3MountPath, ret_record)):
            yield ret_record

def create_dataframe(sqlContext, msgs_rdd, topics):
    """Create DataFrame for specified topics"""
    return sqlContext.createDataFrame(
        msgs_rdd \
        .filter(lambda (_1, _2, topic): topic in operator.itemgetter(*topics)(kWantedChannels)) \
        .map(lambda x: Row(target=x[0], time=x[1], topic=x[2])))

def get_initial_pose(frames):
    """From all the to-be-searialized frames get the smallest pose for usage of initial point"""
    initial_pose_time = sys.float_info.max 
    initial_pose_msg = None
    for key in frames:
        for msg in frames[key]:
            if msg.topic == kWantedChannels['pose'] and msg.timestamp < initial_pose_time:
                initial_pose_time = msg.timestamp
                initial_pose_msg = msg
    return initial_pose_msg

def construct_frames(frames):
    """Construct the frame by using given messages."""
    """Read the according message ONCE again, to avoid playing big messages in memory"""
    target_dir, msgs = frames
    all_frames_msgs = list(msgs)
    '''
    target_dir looks like:
    'modules/data/labeling/2019/2019-01-03/2019-01-03-14-56-05/20181113152909#SS0

    all_frames_msgs like:
    (
      ('lidar-1-topic','lidar-1-time'), ('camera-1-topic','camera-1-time')...
      ('lidar-2-topic','lidar-2-time'), ('camera-2-topic','camera-2-time')...
      ...
    )
    '''
    record_files = target_partition_to_records(target_dir)
    messages_to_write = defaultdict(list)
    
    for record_file in record_files:
        freader = record.RecordReader(os.path.join(s3_utils.S3MountPath, record_file))
        for msg in freader.read_messages():
            for frame in all_frames_msgs:
                msg_map = dict(frame)
                if msg.topic in msg_map and msg.timestamp == msg_map[msg.topic]:
                    messages_to_write[frame[0][1]].append(msg)
                    break

    initial_pose = get_initial_pose(messages_to_write)
    populator = FramePopulator(target_dir, initial_pose)
    for frame_file in messages_to_write:
        populator.construct_frames(messages_to_write[frame_file])        

def get_sql_query(sqlContext, msgs_rdd):
    lidar_df = create_dataframe(sqlContext, msgs_rdd, ('lidar-128',))
    other_sensor_df = create_dataframe(sqlContext, msgs_rdd, [x for x in kWantedChannels.keys() if x != 'lidar-128'])

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
            SELECT target, time, topic, btopic, MIN(ABS(time-btime)) as mindiff FROM %(table_name)s 
            GROUP BY target, time, topic, btopic
        ) B on A.target=B.target AND A.time=B.time AND A.topic=B.topic
        WHERE ABS(A.time-A.btime)=B.mindiff
        """%{
                'table_name': table_name
            })

def mark_complete(todo_files):
    """Create COMPLETE file to mark the job done"""
    todo_files \
        .keys() \
        .distinct() \
        .map(lambda path: os.path.join(s3_utils.S3MountPath, os.path.join(path, 'COMPLETE'))) \
        .map(os.mknod) \
        .count()

def Main():
    """Main function"""
    #todo_files = get_todo_files()

    sc = spark_utils.GetContext()
    todo_files = sc.parallelize([
        ('modules/data/labeling/2019/2019-01-03/2019-01-03-14-56-05', 'public-test/2019/2019-01-03/2019-01-03-14-56-05/20190103145605.record.00000')
    ])

    msgs_rdd = (todo_files                                                       # -> (target_dir, record_file) 
        .map(record_to_target_partition)                                         # -> (target_dir_partition, record_file)
        .flatMapValues(record_utils.ReadRecord(kWantedChannels.values()))        # -> (target_dir_partition, message)
        .map(lambda (target, msg): (target, msg.timestamp, msg.topic))           # -> (target_dir_partition, timestamp, topic)
        .cache())

    # Transform RDD to DataFrame, run SQL and tranform back when done
    sqlContext = SQLContext(sc)
    sql_query = get_sql_query(sqlContext, msgs_rdd)
    (sqlContext
        .sql(sql_query)                                                         # -> (target, lidar_topic, lidar_time, other_topic, MINIMAL_time) 
        .rdd
        .map(list)                                                             
        .keyBy(operator.itemgetter(0))                                          # -> (target, (lidar_topic, lidar_time, other_topic, MINIMAL_time))
        .mapValues(lambda x: ((x[1], x[2]), (x[3], x[4])))                      # -> (target, ((lidar_topic, lidar_time),(other_topic, MINIMAL_time))
        .groupByKey()                                                           # -> <target, {((lidar_topic, lidar_time),(other_topic, MINIMAL_time)...)}
        .map(construct_frames)                                                  # -> process every partition, with all frames belonging to it
        .count())                                                               # -> simply trigger action

    print('All Done: labeling')

if __name__ == '__main__':
    Main()
