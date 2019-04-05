#!/usr/bin/env python
"""This is a module to extraction features from records with folder path as part of the key"""

from collections import Counter
import operator
import os

import pyspark_utils.op as spark_op

import common.proto_utils as proto_utils

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_rdd_utils import record_to_msgs_rdd
from modules.data.fuel.fueling.control.proto.calibration_table_pb2 import CalibrationTable
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
WANTED_VEHICLE = calibration_table_utils.CALIBRATION_TABLE_CONF.vehicle_type
MIN_MSG_PER_SEGMENT = 1
MARKER = 'CompleteCalibrationTable'


class CalibrationTableFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_feature_extraction')

    def run_test(self):
        """Run test."""
        origin_prefix = 'modules/data/fuel/testdata/control/sourceData'
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'CalibrationTable')
        throttle_train_target_prefix = os.path.join(target_prefix, 'throttle', 'train')
        root_dir = '/apollo'

        list_func = (lambda path: self.get_spark_context().parallelize(
            dir_utils.list_end_files(os.path.join(root_dir, path))))
        # RDD(record_dir)
        todo_tasks = (
            dir_utils.get_todo_tasks(
                origin_prefix, throttle_train_target_prefix, list_func, '', '/' + MARKER))

        glog.info('todo_folders: {}'.format(todo_tasks.collect()))

        dir_to_records = (
            # PairRDD(record_dir, record_dir)
            todo_tasks
            # PairRDD(record_dir, all_files)
            .flatMap(dir_utils.list_end_files)
            # PairRDD(record_dir, record_files)
            .filter(record_utils.is_record_file)
            # PairRDD(record_dir, record_files)
            .keyBy(os.path.dirname))

        glog.info('todo_files: {}'.format(dir_to_records.collect()))

        self.run(dir_to_records, origin_prefix, target_prefix, throttle_train_target_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = os.path.join('modules/control/CalibrationTable/', WANTED_VEHICLE)
        throttle_train_target_prefix = os.path.join(target_prefix, 'throttle', 'train')
        root_dir = s3_utils.S3_MOUNT_PATH
        
        # PairRDD(record_dir, record_files)
        todo_records = (
            dir_utils.get_todo_tasks_prod(origin_prefix, target_prefix, root_dir, bucket, MARKER))
        self.run(todo_records, origin_prefix, target_prefix, throttle_train_target_prefix)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, throttle_train_target_prefix):
        """ processing RDD """
        # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
        valid_msgs = (feature_extraction_rdd_utils
                      .record_to_msgs_rdd(dir_to_records_rdd,
                                          WANTED_VEHICLE, channels, MIN_MSG_PER_SEGMENT, MARKER)
                      .cache())

        # PairRDD((dir_segment, segment_id), (chassis_list, pose_list))
        parsed_msgs = feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(
            valid_msgs)

        calibration_table_rdd = (
            # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
            parsed_msgs
            # PairRDD((dir_segment, segment_id), paired_chassis_msg_pose_msg)
            .mapValues(feature_extraction_utils.pair_cs_pose)
            # PairRDD((dir_segment, segment_id), features)
            .mapValues(calibration_table_utils.feature_generate)
            # PairRDD ((dir_segment, segment_id), filtered_features)
            .mapValues(calibration_table_utils.feature_filter)
            # PairRDD ((dir_segment, segment_id), cutted_features)
            .mapValues(calibration_table_utils.feature_cut)
            # PairRDD ((dir_segment, segment_id), (grid_dict,cutted_features))
            .mapValues(calibration_table_utils.feature_distribute)
            # PairRDD ((dir_segment, segment_id), one_matrix)
            .mapValues(calibration_table_utils.feature_store)
            # RDD(feature_numbers)
            .map(lambda elem: calibration_table_utils.write_h5_train_test
                 (elem, origin_prefix, target_prefix, WANTED_VEHICLE)))

        glog.info('Finished %d calibration_table_rdd!' %
                  calibration_table_rdd.count())

        # RDD (dir_segment)
        (feature_extraction_rdd_utils.mark_complete(valid_msgs, origin_prefix,
                                                    throttle_train_target_prefix, MARKER)
         .count())


if __name__ == '__main__':
    CalibrationTableFeatureExtraction().main()
