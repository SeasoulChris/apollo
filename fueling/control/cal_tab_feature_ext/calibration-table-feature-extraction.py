#!/usr/bin/env python

"""
This is a module to extraction features from records
with folder path as part of the key
"""

from collections import Counter
import operator
import os

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.colored_glog as glog
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils

WANTED_VEHICLE = 'Transit'
MIN_MSG_PER_SEGMENT = 1


class CalibrationTableFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'calibration_table_feature_extraction')

    def run_test(self):
        """Run test."""
        records = ['modules/data/fuel/testdata/control/transit/1.record.00000',
                   'modules/data/fuel/testdata/control/mkz7_s/20190222134935.record.00000',
                   'modules/data/fuel/testdata/control/mkz7_s/20190222134935.record.00001',
                   'modules/data/fuel/testdata/control/mkz7_s/20190222134935.record.00002',
                   'modules/data/fuel/testdata/control/mkz7_s/20190222134935.record.00003',
                   'modules/data/fuel/testdata/control/mkz7_s/20190222134935.record.00004']

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = 'modules/data/fuel/testdata/control/generated'
        root_dir = '/apollo'
        dir_to_records = self.get_spark_context().parallelize(
            records).keyBy(os.path.dirname)

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
        dir_to_records = files.filter(
            record_utils.is_record_file).keyBy(os.path.dirname)
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """
        # -> (dir, record), in absolute path
        dir_to_records = dir_to_records_rdd.map(lambda x: (os.path.join(root_dir, x[0]),
                                                           os.path.join(root_dir, x[1]))).cache()

        selected_vehicles = (
            # -> (dir, vehicle)
            feature_extraction_utils.get_vehicle_of_dirs(dir_to_records)
            # -> (dir, vehicle), where vehicle is WANTED_VEHICLE
            .filter(spark_op.filter_value(lambda vehicle: vehicle == WANTED_VEHICLE))
            # -> dir
            .keys())

        channels = {record_utils.CHASSIS_CHANNEL,
                    record_utils.LOCALIZATION_CHANNEL}
        dir_to_msgs = (
            spark_op.filter_keys(dir_to_records, selected_vehicles)
            # -> (dir, msg)
            .flatMapValues(record_utils.read_record(channels))
            # -> (dir_segment, msg)
            .map(feature_extraction_utils.gen_pre_segment)
            .cache())

        valid_segments = (
            dir_to_msgs
            # -> (dir_segment, topic_counter)
            .mapValues(lambda msg: Counter([msg.topic]))
            # -> (dir_segment, topic_counter)
            .reduceByKey(operator.add)
            # -> (dir_segment, topic_counter)
            .filter(spark_op.filter_value(
                lambda counter:
                    counter.get(record_utils.CHASSIS_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT and
                    counter.get(record_utils.LOCALIZATION_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT))
            # -> dir_segment
            .keys())

        data_rdd = (
            # (dir_segment, msg)
            spark_op.filter_keys(dir_to_msgs, valid_segments)
            # -> (dir_segment, msgs)
            .groupByKey()
            # -> (dir_segment, proto_dict)
            .mapValues(record_utils.messages_to_proto_dict())
            # -> (dir_segment, (chassis_list, pose_list))
            .mapValues(lambda proto_dict: (proto_dict[record_utils.CHASSIS_CHANNEL],
                                           proto_dict[record_utils.LOCALIZATION_CHANNEL]))
            # -> (dir_segment, (chassis, pose))
            .mapValues(feature_extraction_utils.pair_cs_pose))

        # ((folder, time/min), feature_matrix)
        calibration_table_rdd = (data_rdd
                                 # feature generator
                                 .mapValues(calibration_table_utils.feature_generate)
                                 # process feature: feature filter
                                 .mapValues(calibration_table_utils.feature_filter)
                                 # process feature: feature cut
                                 .mapValues(calibration_table_utils.feature_cut)
                                 # process feature: feature distribute
                                 .mapValues(calibration_table_utils.feature_distribute)
                                 # process feature: feature store
                                 .mapValues(calibration_table_utils.feature_store)
                                 # write features to hdf5 files
                                 .map(lambda elem: calibration_table_utils.write_h5_train_test
                                      (elem, origin_prefix, target_prefix, WANTED_VEHICLE)))

        glog.info('Finished %d calibration_table_rdd!' %
                  calibration_table_rdd.count())


if __name__ == '__main__':
    CalibrationTableFeatureExtraction().run_prod()
