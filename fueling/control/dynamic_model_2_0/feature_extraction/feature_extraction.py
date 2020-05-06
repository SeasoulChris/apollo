#!/usr/bin/python

import glob
import os

import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config as config
from fueling.control.dynamic_model_2_0.feature_extraction.interpolation_message import \
     InterPolationMessage
from fueling.control.features.feature_extraction_utils import gen_data_point
from fueling.control.features.feature_extraction_utils import pair_cs_pose
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

# The exact length of chasis messages in a group
SEGMENT_LEN = int(config['DELTA_T'] / config['delta_t'])


def add_valid_groups(valid_groups, group):
    """Check the group and split/add it into groups if it's valid"""
    if len(group) < SEGMENT_LEN:
        return
    sub_groups = [group[idx : idx + SEGMENT_LEN]
                  for idx in range(0, len(group), len(group) - config['SEGMENT_OVERLAP'])]
    for sub_group in sub_groups:
        if len(sub_group) == SEGMENT_LEN:
            valid_groups.append(sub_group)


def group_task_messages(task_messages):
    """Match chasis and pose messages and generate groups"""
    task, messages = task_messages
    cur_interp_msg, cur_pose_msg, iterp_messages = InterPolationMessage(), None, []
    logging.info(F'task: {task}, messages len: {len(messages)}')

    # Loop once to get all the interpolate messages
    for message in messages:
        topic, message_proto = message
        if topic == record_utils.CHASSIS_CHANNEL:
            iterp_messages.append(cur_interp_msg)
            cur_interp_msg = InterPolationMessage(message_proto)
        else:
            cur_pose_msg = message_proto
        cur_interp_msg.add_pose(cur_pose_msg) 

    # Optional, get the invalid points
    invalid_interp_pos = [i for i, x in enumerate(iterp_messages) if not x.is_valid()]
    logging.info(F'invalid points len: {len(invalid_interp_pos)}')

    # Generate valid groups potentially splitted by the invalid points
    last_pos, valid_groups = 0, []
    for pos in invalid_interp_pos:
        if last_pos < pos:
            add_valid_groups(valid_groups, iterp_messages[last_pos : pos])
        last_pos = pos + 1 
    if last_pos < len(iterp_messages):
        add_valid_groups(valid_groups, iterp_messages[last_pos:])
    logging.info(F'valid groups len: {len(valid_groups)}')

    return [(task, group_id, group) for group_id, group in enumerate(valid_groups)]


def generate_dataset(group):
    """Generate dataset based on interpolation objects"""
    # TODO(longtao): do interpolation and extract features here
    return None


def write_segment(output_data_path, task_id_group):
    """Write current data list to hdf5 file"""
    task, group_id, group = task_id_group
    output_data_path = os.path.join(output_data_path, task)
    data_set = generate_dataset(group)
    if not data_set:
        logging.info(F'no dataset generated for group {group_id} and folder {output_data_path}')
        return
    h5_utils.write_h5_single_segment(data_set, output_data_path, group_id)
    logging.info(F'written group {group_id} into folder {output_data_path}')


class FeatureExtraction(BasePipeline):

    def run(self):
        """Run."""
        input_data_path = (self.FLAGS.get('input_data_path') or
            'modules/control/data/records/Mkz7/2019-06-04')
        output_data_path = self.our_storage().abs_path(self.FLAGS.get('output_data_path') or
            'modules/control/dynamic_model_2_0/features/2019-06-04')

        logging.info(F'input_data_path: {input_data_path}, output_data_path: {output_data_path}')

        task_msgs_rdd = spark_helper.cache_and_log('task_msgs_rdd',
            # RDD(files)
            self.to_rdd(self.our_storage().list_files(input_data_path))
            # RDD(record files)
            .filter(record_utils.is_record_file)
            # PairRDD(task, record file)
            .keyBy(os.path.dirname)
            # PairRDD(task, message)
            .flatMapValues(record_utils.read_record([record_utils.CHASSIS_CHANNEL,
                                                     record_utils.LOCALIZATION_CHANNEL]))
            # PairRDD(task, (topic, proto))
            .mapValues(lambda value: (value.topic, record_utils.message_to_proto(value)))
            # PariRDD(task, (topic, proto)), sort by proto.header.timestamp_sec
            .sortBy(lambda x: x[1][1].header.timestamp_sec)
            # PairRDD(task, (sorted messages))
            .groupByKey())

        groups = spark_helper.cache_and_log('FilteredGroups',
            # PairRDD(task, (sorted messages))
            task_msgs_rdd
            # PairRDD(task, group_id, InterPolationMessages as a group)
            .flatMap(group_task_messages))
        
        groups.foreach(lambda task_id_group: write_segment(output_data_path, task_id_group))

        logging.info(F'extracted features to target dir {output_data_path}')


if __name__ == '__main__':
    FeatureExtraction().main()
