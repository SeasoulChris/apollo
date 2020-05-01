#!/usr/bin/python

import glob
import os

import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_utils import gen_data_point
from fueling.control.features.feature_extraction_utils import pair_cs_pose
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

SEGMENT_LEN = 100 * 2
# 90% overlaping
SEGMENT_INTERVAL = 10 * 2
FINAL_SEGMENT_LEN = 100


def write_segment(output_data_path, group):
    """Write current data list to hdf5 file"""
    group_id, data_set = group
    h5_utils.write_h5_single_segment(data_set, output_data_path, group_id)
    logging.info(F'written group {group_id} into folder {output_data_path}')

def count_messages_filter(messages_rdd, topic):
    """Count messages from chassis and localization topic"""
    return messages_rdd.filter(lambda message: message.topic == topic).count() >= SEGMENT_LEN // 2

def split_rdd_into_groups(rdd, group_size, overlapping):
    """Split RDD into groups with given overlapping"""
    """([1,2,3,4,5], 3, 1) -> ([1,2,3], [3,4,5])"""

    def slide_window(item_idx, window_size):
        """Split RDD evenly into windows with given size"""
        item, idx = item_idx
        return [(idx - offset, (idx, item)) for offset in range(window_size)]

    step = group_size - overlapping
    return (rdd
        # RDD(item, idx)
        .zipWithIndex() 
        # RDD((item, idx)s)
        .flatMap(lambda item_idx: slide_window(item_idx, group_size))
        # PairRDD(key, (idx, item)), apply overlapping
        .filter(lambda x: x[0] % step == 0)
        # PairRDD(key, (idx, item)s)
        .groupByKey()
        # PairRDD(key, (item)s)
        .mapValues(lambda vals: [item for (idx, item) in sorted(vals)]) 
        # PairRDD(key, (item)s)
        .sortByKey()
        # RDD(groups)
        .values()
        # RDD(groups)
        .filter(lambda group: len(group) == group_size))

def get_datapoints(elem):
    """Generate data points from localization and chassis"""
    (chassis, pose_pre) = elem
    pose = pose_pre.pose
    time_stamp = chassis.header.timestamp_sec
    data_point = gen_data_point(pose, chassis)
    # added time as a dimension
    return np.hstack((data_point, time_stamp / 10**9))


class FeatureExtraction(BasePipeline):

    def run(self):
        """Run."""
        input_data_path = (self.FLAGS.get('input_data_path') or 
            'modules/control/data/records/Mkz7/2019-06-04')
        output_data_path = self.our_storage().abs_path(self.FLAGS.get('output_data_path') or
            'modules/control/dynamic_model_2_0/features/2019-06-04')

        logging.info(F'input_data_path: {input_data_path}, output_data_path: {output_data_path}')

        dir_msgs_rdd = (
            # RDD(files)
            self.to_rdd(self.our_storage().list_files(input_data_path))
            # RDD(record files)
            .filter(record_utils.is_record_file)
            # RDD(message), control and chassis message
            .flatMap(record_utils.read_record([record_utils.CHASSIS_CHANNEL,
                                               record_utils.LOCALIZATION_CHANNEL]))
            .sortBy(lambda message: message.timestamp))

        # Check if qualified for extraction 
        if (not count_messages_filter(dir_msgs_rdd, record_utils.CHASSIS_CHANNEL) or
            not count_messages_filter(dir_msgs_rdd, record_utils.LOCALIZATION_CHANNEL)):
            logging.info('not qualified for extraction, quit')
            return

        # RDD((message)s), group of messages
        (split_rdd_into_groups(dir_msgs_rdd, SEGMENT_LEN, SEGMENT_INTERVAL) 
            # PairRDD((messages)s, group_id)
            .zipWithIndex()
            # PairRDD(group_id, (messages)s)
            .map(lambda x: (x[1], x[0]))
            # PairRDD(group_id, (proto)s)
            .mapValues(record_utils.messages_to_proto_dict())
            # PairRDD(group_id, (chassis_list, pose_list)s)
            .mapValues(lambda proto_dict: (proto_dict[record_utils.CHASSIS_CHANNEL],
                                           proto_dict[record_utils.LOCALIZATION_CHANNEL]))
            # PairRDD(group_id, a single message),
            .flatMapValues(pair_cs_pose)
            # PairRDD(group_id, a data point),
            .mapValues(get_datapoints)
            # PairRDD(group_id, (data_point)s)
            .groupByKey()
            # PairRDD(group_id, list of data_points)
            .mapValues(list)
            # Write each group 
            .foreach(lambda group: write_segment(output_data_path, group)))
 
        logging.info(F'extracted features to target dir {output_data_path}')
        

if __name__ == '__main__':
    FeatureExtraction().main()
