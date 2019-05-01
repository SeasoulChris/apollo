#!/usr/bin/env python
""" extracting sample set """
# pylint: disable = fixme
# pylint: disable = no-member

import glob
import os

import colored_glog as glog
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
WANTED_VEHICLE = feature_extraction_utils.FEATURE_KEY.vehicle_type
MIN_MSG_PER_SEGMENT = 100
MARKER = 'CompleteSampleSet'


class SampleSetFeatureExtraction(BasePipeline):
    """ Generate sample set feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set_feature_extraction')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/records'
        target_prefix = os.path.join('/apollo/modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'SampleSet')
        # RDD(record_dirs)
        todo_tasks = (self.to_rdd([origin_prefix])
                      .flatMap(lambda path: glob.glob(os.path.join(path, '*'))))
        print(todo_tasks.first())
        # PairRDD(record_dirs, record_files)
        todo_records = spark_helper.cache_and_log('todo_records',
                                                  dir_utils.get_todo_records(todo_tasks))

        self.run(todo_records, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        origin_prefix = 'modules/control/records/Mkz7'
        target_prefix = os.path.join('modules/control/data/results', 'SampleSet', WANTED_VEHICLE)
        # RDD(record_dirs)
        """ get to do jobs """
        print('end_files:', self.bos().list_files(origin_prefix))
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            # RDD(relative_path_to_vehicle_type)
            self.to_rdd(self.bos().list_files(origin_prefix))
            # RDD('COMPLETE'_files)
            # .filter(lambda path: path.endswith('COMPLETE'))
            # RDD(absolute_path_to_'COMPLETE')
            .mapValues(os.path.dirname))

        print('todo_jobs: ', todo_task_dirs.collect())
        return
        todo_tasks = dir_utils.get_todo_tasks(origin_prefix, target_prefix, 'COMPLETE', MARKER)
        # PairRDD(record_dirs, record_files)
        todo_records = spark_helper.cache_and_log('todo_records',
                                                  dir_utils.get_todo_records(todo_tasks))
        self.run(todo_records, origin_prefix, target_prefix)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix):
        """ processing RDD """
        # RDD(aboslute_dir) which include records of the wanted vehicle
        selected_vehicles = spark_helper.cache_and_log(
            'SelectedVehicles',
            feature_extraction_rdd_utils.wanted_vehicle_rdd(dir_to_records_rdd, WANTED_VEHICLE))

        # PairRDD((dir, timestamp_per_min), msg)
        dir_to_msgs = spark_helper.cache_and_log(
            'DirToMsgs',
            feature_extraction_rdd_utils.msg_rdd(dir_to_records_rdd, selected_vehicles, channels))

        # RDD(dir, timestamp_per_min)
        valid_segments = spark_helper.cache_and_log(
            'ValidSegments',
            feature_extraction_rdd_utils.
            chassis_localization_segment_rdd(dir_to_msgs, MIN_MSG_PER_SEGMENT))

        # PairRDD((dir_segment, segment_id), msg)
        valid_msgs = spark_helper.cache_and_log(
            'ValidMsg',
            feature_extraction_rdd_utils. valid_msg_rdd(dir_to_msgs, valid_segments))

        data_segment_rdd = spark_helper.cache_and_log(
            'DataSegments',
            # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(
                valid_msgs)
            # PairRDD((dir_segment, segment_id), paired_chassis_msg_pose_msg)
            .flatMapValues(feature_extraction_utils.pair_cs_pose)
            # PairRDD((dir, timestamp_sec), data_point)
            .map(feature_extraction_utils.get_data_point)
            # PairRDD((dir, feature_key), (timestamp_sec, data_point))
            .map(feature_extraction_utils.feature_key_value)
            # PairRDD((dir, feature_key), (timestamp_sec, data_point) RDD)
            .groupByKey()
            # PairRDD((dir, feature_key), list of (timestamp_sec, data_point))
            .mapValues(list)
            # # PairRDD((dir, feature_key), one segment)
            .flatMapValues(feature_extraction_utils.gen_segment))

        spark_helper.cache_and_log(
            'H5ResultMarkers',
            # PairRDD((dir, feature_key), one segment)
            data_segment_rdd
            # PairRDD(dir, feature_key), write all segment into a hdf5 file
            .map(lambda elem: feature_extraction_utils.write_segment_with_key(
                elem, origin_prefix, target_prefix))
            # RDD(dir)
            .keys()
            # RDD(dir), which is unique
            .distinct()
            # RDD(MARKER files)
            .map(lambda path: os.path.join(path, MARKER))
            # RDD(MARKER files)
            .map(file_utils.touch))


if __name__ == '__main__':
    SampleSetFeatureExtraction().main()
