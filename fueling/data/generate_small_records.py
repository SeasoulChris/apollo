#!/usr/bin/env python
import datetime
import operator
import os.path

import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.spark_utils as spark_utils


kWantedChannels = {
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

kBucket = 'apollo-platform'
# Original records are public-test/path/to/*.record, sharded to M.
kOriginPrefix = 'public-test/2019/'
# We will process them to small-records/path/to/*.record, sharded to N.
kTargetPrefix = 'small-records/2019/'


def ShardToFile(dir_msg):
    target_dir, msg = dir_msg
    dt = datetime.datetime.fromtimestamp(msg.timestamp / (10**9))
    target_file = os.path.join(target_dir, dt.strftime('%Y-%m-%d-%H-%M-00.record'))
    return target_file, msg

def Main():
    files = s3_utils.ListFiles(kBucket, kOriginPrefix).persist()

    # (task_dir, _), which is "public-test/..." with 'COMPLETE' mark.
    complete_dirs = (files
        .filter(lambda path: path.endswith('/COMPLETE'))
        .keyBy(os.path.dirname))

    # -> target_dir, which is "small-records/..."
    processed_dirs = s3_utils.ListDirs(kBucket, kTargetPrefix).keyBy(lambda path: path)

    (files
        .filter(record_utils.IsRecordFile)       # -> record
        .keyBy(os.path.dirname)                  # -> (task_dir, record)
        .join(complete_dirs)                     # -> (task_dir, (record, _))
        .mapValues(operator.itemgetter(0))       # -> (task_dir, record)
        .map(spark_utils.MapKey(                 # -> (target_dir, record)
             lambda src_dir: src_dir.replace(kOriginPrefix, kTargetPrefix, 1)))
        .subtractByKey(processed_dirs)           # -> (target_dir, record), which is not processed
        .flatMapValues(                          # -> (target_dir, PyBagMessage)
            record_utils.ReadRecord(kWantedChannels))
        .map(ShardToFile)                        # -> (target_file, PyBagMessage)
        .groupByKey()                            # -> (target_file, PyBagMessages)
        .mapValues(                              # -> (target_file, PyBagMessages_sequence)
            lambda msgs: sorted(msgs, key=lambda msg: msg.timestamp))
        .map(spark_utils.MapKey(lambda target_file: os.path.join('/apollo/data/test', os.path.basename(target_file))))  # For test
        .map(record_utils.WriteRecord)           # -> (None)
        .count())                                # Simply trigger action.


if __name__ == '__main__':
    Main()
