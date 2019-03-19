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
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

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
        
        dir_to_records = (
            # PairRDD(dir, record)
            dir_to_records_rdd
            # PairRDD(aboslute_dir, aboslute_path_record)
            .map(lambda x: (os.path.join(root_dir, x[0]), os.path.join(root_dir, x[1])))
            .cache())

        selected_vehicles = (
            # PairRDD(aboslute_dir, vehicle_type)
            feature_extraction_utils.get_vehicle_of_dirs(dir_to_records)
            # PairRDD(aboslute_dir, wanted_vehicle_type)
            .filter(spark_op.filter_value(lambda vehicle: vehicle == WANTED_VEHICLE))
            # RDD(aboslute_dir) which include records of the wanted vehicle
            .keys())

        channels = {record_utils.CHASSIS_CHANNEL,
                    record_utils.LOCALIZATION_CHANNEL}
        dir_to_msgs = (
            # PairRDD(aboslute_path_dir, aboslute_path_record)
            # which include records of the wanted vehicle
            spark_op.filter_keys(dir_to_records, selected_vehicles)
            # PairRDD(dir, msg)
            .flatMapValues(record_utils.read_record(channels))
            # PairRDD(dir_segment, msg)
            .map(feature_extraction_utils.gen_pre_segment)
            .cache())

        valid_segments = (
            dir_to_msgs
            # PairRDD(dir_segment, topic_counter)
            .mapValues(lambda msg: Counter([msg.topic]))
            # PairRDD(dir_segment, topic_counter)
            .reduceByKey(operator.add)
            # PairRDD(dir_segment, topic_counter)
            .filter(spark_op.filter_value(
                lambda counter:
                    counter.get(record_utils.CHASSIS_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT and
                    counter.get(record_utils.LOCALIZATION_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT))
            # RDD(dir_segment)
            .keys())

        data_segment_rdd = (
            # PairRDD((dir, timestamp_per_min), msg)
            spark_op.filter_keys(dir_to_msgs, valid_segments)
            # PairRDD ((dir, timestamp_per_min), msgs)
            .groupByKey()
            # PairRDD((dir, timestamp_per_min), proto_dict)
            .mapValues(record_utils.messages_to_proto_dict())
            # PairRDD((dir, timestamp_per_min), (chassis_list, pose_list))
            .mapValues(lambda proto_dict: (proto_dict[record_utils.CHASSIS_CHANNEL],
                                           proto_dict[record_utils.LOCALIZATION_CHANNEL]))
            # PairRDD(dir,  (paired_pose_chassis))
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
