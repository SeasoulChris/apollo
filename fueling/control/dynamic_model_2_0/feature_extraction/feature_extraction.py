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


def write_segment(elem):
    """Write current data list to hdf5 file"""
    (folder_path, file_name), data_set = elem
    h5_utils.write_h5_single_segment(data_set, folder_path, file_name)

def count_messages_filter(messages_rdd, topic):
    """Count messages from chassis and localization topic"""
    return messages_rdd.filter(lambda message: message.topic == topic).count() >= SEGMENT_LEN // 2

def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    logging.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msgs: msgs.timestamp)
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
                                               record_utils.LOCALIZATION_CHANNEL])))

        # Check if qualified for extraction 
        if (not count_messages_filter(dir_msgs_rdd, record_utils.CHASSIS_CHANNEL) or
            not count_messages_filter(dir_msgs_rdd, record_utils.LOCALIZATION_CHANNEL)):
            logging.info('not qualified for extraction, quit')
            return

        # RDD(messages)
        (dir_msgs_rdd
            # PairRDD(target_dir, message)
            .map(lambda message: (output_data_path, message))
            # PairRDD(target_dir, (messages))
            .groupByKey()
            # PairRDD(target_dir, (segment_id, messages))
            .flatMap(partition_data)
            # PairRDD((target_dir, segment_id), messages)
            .map(lambda args: ((args[0], args[1]), args[2]))
            # PairRDD((target_dir, segment_id), proto_dict)
            .mapValues(record_utils.messages_to_proto_dict())
            # PairRDD((target_dir, segment_id), (chassis_list, pose_list))
            .mapValues(lambda proto_dict: (proto_dict[record_utils.CHASSIS_CHANNEL],
                                           proto_dict[record_utils.LOCALIZATION_CHANNEL]))
            # PairRDD((target_dir, group_id), a single message),
            .flatMapValues(pair_cs_pose)
            # PairRDD((target_dir, group_id), a data point),
            .mapValues(get_datapoints)
            # PairRDD((target_dir, group_id), data_points)
            .groupByKey()
            # PairRDD((target_dir, group_id), list of data_points)
            .mapValues(list)
            # Write each data points
            .foreach(write_segment))
 
        logging.info(F'extracted features to target dir {output_data_path}')
        

if __name__ == '__main__':
    FeatureExtraction().main()
