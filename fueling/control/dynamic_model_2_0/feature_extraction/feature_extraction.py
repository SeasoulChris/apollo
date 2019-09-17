#!/usr/bin/python

import glob
import os

from absl import logging
import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_utils import gen_data_point
from fueling.control.features.feature_extraction_utils import pair_cs_pose
import fueling.common.h5_utils as h5_utils
import fueling.common.record_utils as record_utils
import fueling.common.storage.bos_client as bos_client
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils

SEGMENT_LEN = 100 * 2
SEGMENT_INTERVAL = 10 * 2  # 90% overlaping
FINAL_SEGMENT_LEN = 100
PROD_INPUT_DIR = 'modules/control/data/records/Mkz7/2019-04-25'
PROD_TARGET_DIR = 'modules/control/DM2/test'


def write_segment(elem):
    """Write current data list to hdf5 file"""
    (folder_path, file_name), data_set = elem
    h5_utils.write_h5_single_segment(data_set, folder_path, file_name)
    return folder_path


def count_msgs(dir_msgRDD):
    """Count messages from chassis and localization topic"""
    folder, messages = dir_msgRDD
    count_chassis = 0
    count_localization = 0
    for message in messages:
        if message.topic == record_utils.CHASSIS_CHANNEL:
            count_chassis += 1
        elif message.topic == record_utils.LOCALIZATION_CHANNEL:
            count_localization += 1
    logging.info('{} chassis messages for record folder {}'.format(count_chassis, folder))
    logging.info('{} localization messages for record folder {}'.format(count_localization, folder))
    if count_chassis < SEGMENT_LEN / 2 or count_localization < SEGMENT_LEN / 2:
        return (folder, [])
    return dir_msgRDD


def partition_data(target_msgs, segment_len=SEGMENT_LEN, segment_int=SEGMENT_INTERVAL):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    logging.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msgs: msgs.timestamp)
    msgs_groups = [msgs[idx: idx + segment_len]
                   for idx in range(0, len(msgs), segment_int)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]


def get_datapoints(elem):
    """Generate data points from localization and chassis"""
    (chassis, pose_pre) = elem
    pose = pose_pre.pose
    time_stamp = chassis.header.timestamp_sec
    data_point = gen_data_point(pose, chassis)
    # added time as a dimension
    return np.hstack((data_point, time_stamp / 10**9))


def count_pair_msgs(elem):
    """ count paired msgs """
    (segment_dir, group_id), data_list = elem
    logging.info('{} data points for record folder {} segment {}'.format(
        len(data_list), segment_dir, group_id))


class FeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set')

    def run_test(self):
        """Run test."""
        # test data dir (folder/record_files)
        test_data_dirs = '/apollo/modules/data/fuel/apps/local/DM2'
        all_dirs = glob.glob(os.path.join(test_data_dirs, '*'))
        target_dir = '/apollo/modules/data/fuel/testdata/control/DM2_OUT/0816'
        # RDD(tasks)
        task = self.to_rdd(all_dirs)
        self.run(task, test_data_dirs, target_dir)

    def run_prod(self):
        """Run prod."""
        # get subfolder of all records
        task = self.to_rdd([PROD_INPUT_DIR])
        task = spark_helper.cache_and_log(
            'todo_jobs',
            # RDD(relative_path_to_vehicle_type)
            task
            # RDD(files)
            .flatMap(self.bos().list_files)
            # RDD(folders)
            .map(os.path.dirname)
            # RDD(distinct folders)
            .distinct(), 1)
        self.run(task, PROD_INPUT_DIR, PROD_TARGET_DIR)

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
            .groupByKey(), 0)

        hdf5_dir = spark_helper.cache_and_log(
            'write_datapoint_to_hdf5',
            # PairRDD(valid target_dir, (message)s)
            dir_msgs_rdd
            # PairRDD(valid target_dir, (message)s or [])
            .map(count_msgs)
            # PairRDD(valid target_dir, (message)s)
            .filter(spark_op.filter_value(lambda msgs: len(msgs) > 0))
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
            # PairRDD((target_dir, group_id), data_point RDD)
            .groupByKey()
            # PairRDD((target_dir, group_id), list of data_point)
            .mapValues(list), 0)

        # PairRDD((target_dir, group_id), len of list)
        hdf5_dir_count = hdf5_dir.foreach(count_pair_msgs)

        (hdf5_dir
         # PairRDD((target_dir, group_id), list of data_point)
         .filter(spark_op.filter_value(lambda msgs: len(msgs) == FINAL_SEGMENT_LEN))
         # RDD(hdf5 file dir)
         .map(write_segment)).count()


if __name__ == '__main__':
    FeatureExtraction().main()
