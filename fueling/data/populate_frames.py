#!/usr/bin/env python

import datetime
import operator
import os
import pprint
import re
import time

from cyber_py import record

import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.spark_utils as spark_utils

kBucket = 'apollo-platform'
# Original records are public-test/path/to/*.record, sharded to M.
kOriginPrefix = 'public-test/2019/'
# We will process them to small-records/path/to/*.record, sharded to N.
kTargetPrefix = 'modules/data/labeling/'
# How many records are contained by one slice
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

def get_todo_files(bucket, targetPrefix):
    """Get to be processed files in rdd format."""
    files = s3_utils.ListFiles(kBucket, kOriginPrefix).persist()

    # (task_dir, _), which is "public-test/..." with 'COMPLETE' mark.
    complete_dirs = (files
        .filter(lambda path: path.endswith('/COMPLETE'))
        .keyBy(os.path.dirname))

    # target_dir
    processed_dirs = s3_utils.ListDirs(bucket, targetPrefix).keyBy(lambda path: path)

    # Find all todo jobs.
    todo_files = (files
        .filter(record_utils.IsRecordFile)  # -> record
        .keyBy(os.path.dirname)             # -> (task_dir, record)
        .join(complete_dirs)                # -> (task_dir, (record, _))
        .mapValues(operator.itemgetter(0))  # -> (task_dir, record)
        .map(spark_utils.MapKey(lambda src_dir: src_dir.replace(kOriginPrefix, targetPrefix, 1)))
                                            # -> (target_dir, record)
        .subtractByKey(processed_dirs)      # -> (target_dir, record), which is not processed
        .cache())
    
    return todo_files

def find_closest_msgs(task_record, lidar_msg, related_msgs):
    """Find closest values to lidar message for each topic"""
    closest_msgs = (task_record,) + (lidar_msg,)
    for msg in related_msgs:
        closest_msgs += (min(msg, key=lambda x:abs(float(x[0])-float(lidar_msg[0]))),)
    return closest_msgs

def get_related_topic_msgs(msgs_rdd):
    """Get collections of messages with required topics"""
    related_msgs = []
    for k in kWantedChannels:
        if k == 'lidar-128':
            continue
        related_msgs.append(msgs_rdd.filter(lambda x: x[1] == kWantedChannels[k]).collect())
    return related_msgs

def record_to_target(task_record_pair):
    """Partition source records to slices"""
    task_dir, record = task_record_pair
    record_fileparts = os.path.basename(record).split('.')
    task_dir += '/{}#SS{}'.format(record_fileparts[0], int(record_fileparts[2])/kSliceSize)
    return ((task_dir, record), record)

def target_dir_to_records(target_dir, record_file):
    """Convert target dir to actual record files list"""
    seq = int(re.search(r'[\S]*\d{14}#SS([\d]*)$', target_dir, re.M|re.I).group(1))
    src_record_file = record_file.rsplit('.', 1)[0]
    for rec in range(seq * kSliceSize, (seq+1) * kSliceSize):
        ret_record = '{}.{:05d}'.format(src_record_file, rec)
        if os.path.exists(os.path.join(s3_utils.S3MountPath, ret_record)):
            yield ret_record

def write_to_file(file_path, msgs):
    """TODO: place holder.  Write message to file"""
    with open(file_path, 'w') as outfile:
        for msg in msgs:
            outfile.write('{}-{}'.format(msg.topic, msg.timestamp))

def construct_frames(frames):
    """Construct the frame by using given messages."""
    """TODO: need to change this dirty hard worker"""
    target_dir, msgs = frames
    all_frames_msgs = list(msgs)
    '''
    all_frames_msgs like:
    (
      ('target_dir', 'record_file'),('lidar-1-time', 'lidar-topic'),('camera-1-time', 'camera-topic')...
      ('target_dir', 'record_file'),('lidar-2-time', 'lidar-topic'),('camera-1-time', 'camera-topic')...
      ...
    )
    '''
    record_files = target_dir_to_records(target_dir, all_frames_msgs[0][0][1])
    messages_to_write = {}

    for record_file in record_files:
        freader = record.RecordReader(os.path.join(s3_utils.S3MountPath, record_file))
        for msg in freader.read_messages():
            for frame in all_frames_msgs:
                msg_map = dict(frame[1:])
                if msg.timestamp in msg_map and msg.topic == msg_map[msg.timestamp]:
                    if frame[1][0] not in messages_to_write:
                        messages_to_write[frame[1][0]] = []
                    messages_to_write[frame[1][0]].append(msg)

    for frame_file in messages_to_write:
        file_path = os.path.join(s3_utils.S3MountPath, os.path.join(target_dir, str(frame_file)))
        write_to_file(file_path, messages_to_write[frame_file])            

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
    #todo_files = get_todo_files(kBucket, kTargetPrefix)

    sc = spark_utils.GetContext()
    todo_files = sc.parallelize([
        ('labeling-frames/2019/2019-01-03/2019-01-03-14-56-05', 'apollo/modules/data/fuel/20181113152909.record.00000')
    ])

    msgs_rdd = todo_files \
        .filter(lambda (task_dir, record): record_utils.IsRecordFile(record)) \
        .map(record_to_target) \
        .flatMapValues(record_utils.ReadRecord(kWantedChannels.values())) \
        .map(lambda (task_record, msg): (task_record, (msg.timestamp, msg.topic))) \
        .cache()

    related_msgs = get_related_topic_msgs(msgs_rdd.map(operator.itemgetter(1)))

    msgs_rdd \
        .filter(lambda (task_record, msg): msg[1] == kWantedChannels['lidar-128']) \
        .map(lambda (task_record, msg): find_closest_msgs(task_record, msg, related_msgs)) \
        .groupBy(lambda task_msg_tuple: task_msg_tuple[0][0]) \
        .map(construct_frames) \
        .count()
    
    mark_complete(todo_files)

    print('All Done: labeling')


if __name__ == '__main__':
    Main()
