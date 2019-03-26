#!/usr/bin/env python
""" extracting sample set """
# pylint: disable = fixme
# pylint: disable = no-member

from collections import Counter
import operator
import os

import glob
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.feature_extraction_rdd_utils import record_to_msgs_rdd
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

channels = {record_utils.CHASSIS_CHANNEL,record_utils.LOCALIZATION_CHANNEL}
WANTED_VEHICLE = feature_extraction_utils.FEATURE_KEY.vehicle_type
MIN_MSG_PER_SEGMENT = 100
MARKER = 'CompleteSampleSet'

class SampleSetFeatureExtraction(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set_feature_extraction')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                      WANTED_VEHICLE, 'SampleSet')
        root_dir = '/apollo'

        # complete file is writtern in original folder
        # RDD(record_dir)
        todo_tasks = 
            dir_utils.get_todo_tasks(origin_prefix, target_prefix, 
            lambda path: self.get_spark_context().parallelize(dir_utils.
            list_end_files(os.path.join(root_dir, path))), '', '/' + MARKER)

        # PairRDD(record_dir, record_files)
        dir_to_records = todo_tasks.flatMap(dir_utils.list_end_files).keyBy(os.path.dirname)

        glog.info('todo_files: {}'.format(dir_to_records.collect()))
       
        self.run(dir_to_records, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = os.path.join('modules/control/feature_extraction_hf5/hdf5_training/',
                                     WANTED_VEHICLE, 'SampleSet')
        root_dir = s3_utils.S3_MOUNT_PATH

        # RDD(record_dir)
        todo_tasks = (dir_utils.get_todo_tasks(origin_prefix, target_prefix
                                              lambda path: s3_utils.list_files(bucket, path),
                                              '/COMPLETE', '/' + MARKER)
                      # RDD(record_files)
                      .map(os.listdir)
                      # RDD(absolute_record_files)
                      .map(lambda record_dir: os.path.join(root_dir, record_dir)))

        # PairRDD(record_dir, record_files)
        dir_to_records = todo_tasks.filter(record_utils.is_record_file).keyBy(os.path.dirname)

        self.run(dir_to_records, origin_prefix, target_prefix)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix):
        """ processing RDD """
    
        # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
        parsed_msgs = record_to_msgs_rdd(dir_to_records_rdd, WANTED_VEHICLE, 
                                         channels, MIN_MSG_PER_SEGMENT, MARKER)
   
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
