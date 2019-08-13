import glob
import os

import colored_glog as glog
import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_utils import gen_data_point
from fueling.control.features.feature_extraction_utils import pair_cs_pose
import fueling.common.h5_utils as h5_utils
import fueling.common.record_utils as record_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils

# channels = [record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL]
SEGMENT_LEN = 6000 * 2  # 1 min msgs of chassis and localization
SEGMENT_INTERVAL = 2000 * 2  # 20 secs msgs of chassis and localization
# Maximum allowed time gap betwee two messages
MAX_PHASE_DELTA = 0.01 / 2


def write_segment(elem):
    """Write current data list to hdf5 file"""
    (folder_path, file_name), data_set = elem
    h5_utils.write_h5_single_segment(data_set, folder_path, file_name)
    return folder_path


def count_msgs(dir_msgRDD):
    """Count Msgs from chassis and localization topic"""
    folder, messages = dir_msgRDD
    count_chassis = 0
    count_localization = 0
    for message in messages:
        if message.topic == record_utils.CHASSIS_CHANNEL:
            count_chassis += 1
        elif message.topic == record_utils.LOCALIZATION_CHANNEL:
            count_localization += 1
    glog.info('{} chassis messages for record folder {}'.format(count_chassis, folder))
    glog.info('{} localization messages for record folder {}'.format(count_localization, folder))
    return (folder, (count_chassis, count_localization))
    # return count_chassis > SEGMENT_LEN / 2 and count_localization > SEGMENT_LEN / 2


def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    glog.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msg: msg.timestamp)
    msgs_groups = [msgs[idx: idx + SEGMENT_LEN]
                   for idx in range(0, len(msgs), SEGMENT_INTERVAL)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]


def get_datapoints(elem):
    """Generate data points from localization and chassis"""
    (chassis, pose_pre) = elem
    pose = pose_pre.pose
    time_stamp = chassis.header.timestamp_sec
    data_point = gen_data_point(pose, chassis)
    # added time as a dimension
    return np.hstack((data_point, time_stamp / 10**9))


class FeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set')

    def run_test(self):
        """Run test."""
        # test data dir (folder/record_files)
        test_data_dirs = '/apollo/modules/data/fuel/testdata/control/DM2/2019-04-15-15-38-16_s'
        target_dir = '/apollo/modules/data/fuel/testdata/control/DM2_OUT/2019-04-15-15-38-16_s'
        # RDD(tasks)
        task = self.to_rdd([test_data_dirs])
        self.run(task, test_data_dirs, target_dir)

    def run(self, task, original_prefix, target_prefix):
        # configurable segments
        dir_msgs_rdd = spark_helper.cache_and_log(
            'record_to_msgs',
            # RDD(tasks), with absolute paths
            task
            # PairRDD(target_dir, task), the map of target dirs and source dirs
            .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1))
            # PairRDD(target_dir, record_file)
            .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')))
            # PairRDD(target_dir, record_file), filter out unqualified files
            .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file)))
            # PairRDD(target_dir, message), control and chassis message
            .flatMapValues(record_utils.read_record([record_utils.CHASSIS_CHANNEL,
                                                     record_utils.LOCALIZATION_CHANNEL]))
            # PairRDD(target_dir, (message)s)
            .groupByKey(), 1)

        # RDD(valid dirs)
        valid_dirs = spark_helper.cache_and_log(
            'write_datapoint_to_hdf5',
            dir_msgs_rdd
            # PairRDD(target_dir, (chassis_msg_numbers, localization_msg_numbers))
            .map(count_msgs)
            # PairRDD(valid target_dir, (chassis_msg_numbers, localization_msg_numbers))
            .filter(lambda (_, (count_chassis, count_localization)):
                    count_chassis >= SEGMENT_LEN / 2
                    and count_localization >= SEGMENT_LEN / 2)
            # PairRDD(valid target_dir)
            .keys(), 1)

        # PairRDD(valid target_dir, (message)s)
        valid_msg_dir = spark_op.filter_keys(dir_msgs_rdd, valid_dirs)

        hdf5_dir = spark_helper.cache_and_log(
            'write_datapoint_to_hdf5',
            # PairRDD(valid target_dir, (message)s)
            valid_msg_dir
            # RDD(target_dir, segment_id, group of (message)s), divide messages into groups
            .flatMap(partition_data)
            # PairRDD((target_dir, segment_id), (message)s)
            .map(lambda (target_dir, segment_id, msgs): ((target_dir, segment_id), msgs))
            .mapValues(record_utils.messages_to_proto_dict())
            # # PairRDD((dir_segment, segment_id), (chassis_list, pose_list))
            .mapValues(lambda proto_dict: (proto_dict[record_utils.CHASSIS_CHANNEL],
                                           proto_dict[record_utils.LOCALIZATION_CHANNEL]))
            # PairRDD(target_dir, group_id, a single message),
            .flatMapValues(pair_cs_pose)
            # PairRDD(target_dir, group_id, a data point),
            .mapValues(get_datapoints)
            # PairRDD((vehicle, dir, feature_key), data_point RDD)
            .groupByKey()
            # PairRDD((vehicle, dir, feature_key), list of data_point)
            .mapValues(list)
            # PairRDD((vehicle, dir, feature_key), data_point RDD)
            .map(write_segment), 1)


if __name__ == '__main__':
    FeatureExtraction().main()
