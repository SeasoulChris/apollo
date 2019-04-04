#!/usr/bin/env python
"""This is a module to extraction features from records with folder path as part of the key"""

from collections import Counter
import operator
import os

import pyspark_utils.op as spark_op

import common.proto_utils as proto_utils

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_rdd_utils import record_to_msgs_rdd
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
from modules.data.fuel.fueling.control.proto.calibration_table_pb2 import calibrationTable

channels = {record_utils.CHASSIS_CHANNEL,record_utils.LOCALIZATION_CHANNEL}
# WANTED_VEHICLE = 'Transit'
WANTED_VEHICLE = calibration_table_utils.CALIBRATION_TABLE_CONF.vehicle_type
MIN_MSG_PER_SEGMENT = 1

class CalibrationTableFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_feature_extraction')

    def run_test(self):
        """Run test."""
        records = ['modules/data/fuel/testdata/control/calibration_table/transit/1.record.00000']
        origin_prefix = 'modules/data/fuel/testdata/control/calibration_table/'
        target_prefix = 'modules/data/fuel/testdata/control/calibration_table/generated'
        root_dir = '/apollo'
        # PairRDD(record_dir, record_files)
        dir_to_records = self.get_spark_context().parallelize(records).keyBy(os.path.dirname)

        self.run(dir_to_records, origin_prefix, target_prefix, root_dir)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = 'modules/control/feature_extraction_hf5/2019/'
        root_dir = s3_utils.S3_MOUNT_PATH
        files = s3_utils.list_files(bucket, origin_prefix).cache()
        complete_dirs = files.filter(lambda path: path.endswith('/COMPLETE')).map(os.path.dirname)
        # PairRDD(record_dir, record_files)
        dir_to_records = files.filter(record_utils.is_record_file).keyBy(os.path.dirname)
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """
        target_prefix = os.path.join(root_dir, target_prefix)

        # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
        parsed_msgs = record_to_msgs_rdd(dir_to_records_rdd, root_dir, WANTED_VEHICLE, 
                                          channels, MIN_MSG_PER_SEGMENT)
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

        glog.info('Finished %d calibration_table_rdd!' % calibration_table_rdd.count())


if __name__ == '__main__':
    CalibrationTableFeatureExtraction().main()
