#!/usr/bin/env python
""" extracting sample set for mkz7 """
# pylint: disable = fixme
# pylint: disable = no-member
import os

import glog
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.control.features.common_feature_extraction as CommonFE


class SampleSetFeatureExtraction(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """
    WANTED_VEHICLE = 'Mkz7'

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set_feature_extraction')

    def run_test(self):
        """Run test."""
        records = ["modules/data/fuel/testdata/control/left_40_10/1.record.00000",
                   "modules/data/fuel/testdata/control/transit/1.record.00000"
                   ]

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
        root_dir = s3_utils.S3_MOUNT_PATH
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """
        wanted_chs = ['/apollo/canbus/chassis',
                      '/apollo/localization/pose']

        folder_vehicle_rdd = (
            # (dir, record)
            dir_to_records_rdd
            # -> (dir, record), in absolute path
            .map(lambda x: (os.path.join(root_dir, x[0]), os.path.join(root_dir, x[1])))
            # -> (dir, HMIStatus msg)
            .flatMapValues(record_utils.read_record(['/apollo/hmi/status']))
            # -> (dir, HMIStatus)
            .mapValues(record_utils.message_to_proto)
            # -> (dir, current_vehicle)
            .mapValues(lambda elem: elem.current_vehicle)
            # -> (dir, current_vehicle)
            .filter(spark_op.filter_value(
                lambda vehicle: vehicle == SampleSetFeatureExtraction.WANTED_VEHICLE))
            # -> dir
            .keys()
            # Deduplicate.
            .distinct())

        glog.info('Finished %d folder_vehicle_rdd!' %
                  folder_vehicle_rdd.count())
        glog.info('folder_vehicle_rdd first elem: %s ' %
                  folder_vehicle_rdd.take(1))

        channels_rdd = (
            # (dir)
            folder_vehicle_rdd
            # -> (dir,dir)
            .keyBy(lambda x: x)
            # -> (dir,record)
            .flatMapValues(CommonFE.folder_to_record)
            # ->(dir,pose and chassis msg)
            .flatMapValues(record_utils.read_record(wanted_chs))
            # ->(dir,pose or chassis)
            .mapValues(record_utils.message_to_proto))

        glog.info('Finished %d channels_rdd!' %
                  channels_rdd.count())

        pre_segment_rdd = (
            #(dir, pose or chassis)
            channels_rdd
            # -> ((dir,time_stamp_per_min), pose or chassis)
            .map(CommonFE.gen_key)
            # -> combine to the same key
            .combineByKey(CommonFE.to_list, CommonFE.append, CommonFE.extend))
        glog.info('Finished %d pre_segment_rdd!' %
                  pre_segment_rdd.count())

        data_rdd = (
            # ((dir,time_stamp_per_min), pose and chassis list)
            pre_segment_rdd
            # -> (key, (has_pose_list+has_chassis_list), (pose_list, chassis_list))
            .mapValues(CommonFE.process_seg)
            # -> (key, (has_pose_list+has_chassis_list), (pose_list, chassis_list))
            .filter(lambda elem: elem[1][0] == 2)
            # ->(key,  (pose_list, chassis_list))
            .mapValues(lambda elem: elem[1])
            # ->(key,  (paired_pose_chassis))
            .flatMapValues(CommonFE.pair_cs_pose)
            # ->((dir, time_stamp_sec), data_point)
            .map(CommonFE.get_data_point))

        glog.info('Finished %d data_rdd!' %
                  data_rdd.count())

        # data feature set
        featured_data_rdd = (
            #((dir, time_stamp_sec), data_point)
            data_rdd
            # -> ((dir,feature_key),(time_stamp_sec,data_point))
            .map(CommonFE.feature_key_value)
            # -> ((dir,feature_key), list of (time_stamp_sec,data_point))
            .combineByKey(CommonFE.to_list, CommonFE.append, CommonFE.extend))

        glog.info('Finished %d featured_data_rdd!' % featured_data_rdd.count())

        data_segment_rdd = (featured_data_rdd
                            # generate segment w.r.t time
                            .mapValues(CommonFE.gen_segment)
                            # write all segment into a hdf5 file
                            .map(lambda elem:
                                 CommonFE.write_h5_with_key(elem, origin_prefix, target_prefix, SampleSetFeatureExtraction.WANTED_VEHICLE)))
        glog.info('Finished %d data_segment_rdd!' %
                  data_segment_rdd.count())


if __name__ == '__main__':
    SampleSetFeatureExtraction().run_test()
