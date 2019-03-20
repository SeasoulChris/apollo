#!/usr/bin/env python
""" extracting sample set """
# pylint: disable = fixme
# pylint: disable = no-member

from collections import Counter
import operator
import os

import glob
import glog
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_rdd_utils import record_to_msgs_rdd
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

channels = {record_utils.CHASSIS_CHANNEL,record_utils.LOCALIZATION_CHANNEL}
WANTED_VEHICLE = feature_extraction_utils.FEATURE_KEY.vehicle_type
MIN_MSG_PER_SEGMENT = 100


class SampleSetFeatureExtraction(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set_feature_extraction')

    def run_test(self):
        """Run test."""
        records = [
            "modules/data/fuel/testdata/control/left_40_10/1.record.00000",
            "modules/data/fuel/testdata/control/transit/1.record.00000",
        ]

        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = 'modules/data/fuel/testdata/control/generated'
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
        complete_dirs = files.filter(
            lambda path: path.endswith('/COMPLETE')).map(os.path.dirname)
        # PairRDD(record_dir, record_files)
        dir_to_records = files.filter(record_utils.is_record_file).keyBy(os.path.dirname)
        root_dir = s3_utils.S3_MOUNT_PATH
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """
        
        # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
        parsed_msgs = record_to_msgs_rdd(dir_to_records_rdd, root_dir, WANTED_VEHICLE, 
                                          channels, MIN_MSG_PER_SEGMENT)

        data_segment_rdd = (
            # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
            parsed_msgs
            # PairRDD((dir_segment, segment_id), paired_chassis_msg_pose_msg)
            .flatMapValues(feature_extraction_utils.pair_cs_pose)
            # PairRDD((dir, timestamp_sec), data_point)
            .map(feature_extraction_utils.get_data_point)
            # PairRDD((dir, feature_key), (timestamp_sec, data_point))
            .map(feature_extraction_utils.feature_key_value)
            # PairRDD((dir, feature_key), list of (timestamp_sec, data_point))
            .combineByKey(feature_extraction_utils.to_list, feature_extraction_utils.append,
                          feature_extraction_utils.extend)
            # PairRDD((dir, feature_key), segments)
            .mapValues(feature_extraction_utils.gen_segment)
            # RDD(dir, feature_key), write all segment into a hdf5 file
            .map(lambda elem: feature_extraction_utils.write_h5_with_key(
                elem, origin_prefix, target_prefix, WANTED_VEHICLE)))
        glog.info('Finished %d data_segment_rdd!' % data_segment_rdd.count())


if __name__ == '__main__':
    SampleSetFeatureExtraction().run_test()
