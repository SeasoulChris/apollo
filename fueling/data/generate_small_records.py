#!/usr/bin/env python

import os.path

from cyber_py.record import RecordReader

import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


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
kOriginKeyPrefix = 'public-test/2019/'
# We will process them to small-records/path/to/*.record, sharded to N.
kTargetKeyPrefix = 'small-records/2019/'

FuncKeyToPath = lambda key: '/mnt/bos/%s' % key


def Main():
    files = s3_utils.ListFiles(kBucket, kOriginKeyPrefix)
    files.persist()

    upload_complete_dirs = (files
        .filter(lambda path: path.endswith('/UPLOAD_COMPLETE'))
        .keyBy(os.path.dirname))

    # -> target_dir, which is "small-records/..."
    processed_dirs = s3_utils.ListDirs(kBucket, kTargetKeyPrefix).map(lambda path: path, None)

    (files
        .filter(record_utils.IsRecordFile)       # -> record
        .keyBy(os.path.dirname)                  # -> (task_dir, record)
        .join(upload_complete_dirs)              # -> (task_dir, (record, todo=True))
        .mapValues(lambda path, _: path)         # -> (task_dir, record)
        # -> (target_dir, record)
        .map(lambda task_dir, record: task_dir.replace(kOriginKeyPrefix, kTargetKeyPrefix, 1), record)
        .subtractByKey(processed_dirs)           # -> (target_dir, file), which is not processed
        .flatMapValues(record_utils.ReadRecord)  # -> (target_dir, PyBagMessage)
        .groupByKey()                            # -> (target_dir, PyBagMessages)
        .map(record_utils.WriteRecord)           # -> (target_dir, None)
        .count())                                # Simply trigger action.


if __name__ == '__main__':
    Main()
