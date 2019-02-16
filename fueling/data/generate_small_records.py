#!/usr/bin/env python
import datetime
import operator
import os

import glog
import pyspark_utils.op as spark_op

import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


WANTED_CHANNELS = {
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

BUCKET = 'apollo-platform'
# Original records are public-test/path/to/*.record, sharded to M.
ORIGIN_PREFIX = 'public-test/2019/'
# We will process them to small-records/path/to/*.record, sharded to N.
TARGET_PREFIX = 'small-records/2019/'


def ShardToFile(dir_msg):
    target_dir, msg = dir_msg
    dt = datetime.datetime.fromtimestamp(msg.timestamp / (10**9))
    target_file = os.path.join(target_dir, dt.strftime('%Y-%m-%d-%H-%M-00.record'))
    return target_file, msg

def Main():
    files = s3_utils.list_files(BUCKET, ORIGIN_PREFIX).cache()

    # (task_dir, _), which is "public-test/..." with 'COMPLETE' mark.
    complete_dirs = (files
        .filter(lambda path: path.endswith('/COMPLETE'))
        .keyBy(os.path.dirname))

    # (target_dir, _), which is "small-records/..."
    processed_dirs = s3_utils.list_dirs(BUCKET, TARGET_PREFIX).map(lambda path: path, None)

    # Find all todo jobs.
    todo_jobs = (files
        .filter(record_utils.is_record_file)  # -> record
        .keyBy(os.path.dirname)               # -> (task_dir, record)
        .join(complete_dirs)                  # -> (task_dir, (record, _))
        .mapValues(operator.itemgetter(0))    # -> (task_dir, record)
        .map(spark_op.do_key(lambda src_dir: src_dir.replace(ORIGIN_PREFIX, TARGET_PREFIX, 1)))
                                              # -> (target_dir, record)
        .subtractByKey(processed_dirs)        # -> (target_dir, record), which is not processed
        .cache())

    # Read the input data and write to target file.
    records_count = (todo_jobs
        .flatMapValues(record_utils.read_record(WANTED_CHANNELS))  # -> (target_dir, PyBagMessage)
        .map(ShardToFile)                                          # -> (target_file, PyBagMessage)
        .groupByKey()                                              # -> (target_file, PyBagMessages)
        .mapValues(lambda msgs: sorted(msgs, key=lambda msg: msg.timestamp))
                                         # -> (target_file, PyBagMessages_sequence)
        .map(record_utils.write_record)  # -> (None)
        .count())                        # Simply trigger action.
    glog.info('Finished %d records!' % records_count)

    # Create COMPLETE mark.
    tasks_count = (todo_jobs
        .keys()                                            # -> target_dir
        .distinct()                                        # -> unique_target_dir
        .map(lambda path: os.path.join(s3_utils.S3_MOUNT_PATH, path, 'COMPLETE'))
                                                           # -> unique_target_dir/COMPLETE
        .map(os.mknod)                                     # Touch file
        .count())                                          # Simply trigger action.
    glog.info('Finished %d tasks!' % tasks_count)


if __name__ == '__main__':
    Main()
