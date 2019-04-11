#!/usr/bin/env python
""" extracting sample set """
# pylint: disable = fixme
# pylint: disable = no-member

import glob
import os

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils


channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
WANTED_VEHICLE = feature_extraction_utils.FEATURE_KEY.vehicle_type
MIN_MSG_PER_SEGMENT = 10
MARKER = 'CompleteSampleSet'


class SampleSetFeatureExtraction(BasePipeline):
    """ Generate sample set feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set_feature_extraction')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)


        origin_prefix = 'modules/data/fuel/testdata/control/sourceData/SAMPLE_SET'
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'SampleSet')
        root_dir = '/apollo'

        list_func = (lambda path: self.context().parallelize(
            dir_utils.list_end_files(os.path.join(root_dir, path))))
        # RDD(record_dir)
        todo_tasks = (dir_utils
                      .get_todo_tasks(origin_prefix, target_prefix, list_func, '', '/' + MARKER))
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
        self.run(dir_to_records, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = os.path.join(
            'modules/control/feature_extraction_hf5/hdf5_training', WANTED_VEHICLE, 'SampleSet')
        root_dir = s3_utils.S3_MOUNT_PATH
        to_abs_path = False

        list_func = (lambda path: s3_utils.list_files(bucket, path, to_abs_path))

        # RDD(record_dir)
        todo_tasks = (
            # RDD(record_dir)
            dir_utils.get_todo_tasks(
                origin_prefix, target_prefix, list_func, '/COMPLETE', '/' + MARKER)
            # RDD(abs_record_dir)
            .map(lambda record_dir: os.path.join(root_dir, record_dir))
            # PairRDD(record_dir, record_dir)
            .keyBy(lambda record_dir: record_dir)
            # PairRDD(record_dir, record_files)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*record*')))
            .cache())

        task_count = todo_tasks.count()
        glog.info("Get {} TODO tasks: [{}, ...]".format(
            task_count, todo_tasks.first() if task_count > 0 else 'None'))

        self.run(todo_tasks, origin_prefix, target_prefix)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix):
        """ processing RDD """
        # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
        valid_msgs = feature_extraction_rdd_utils.record_to_msgs_rdd(
            dir_to_records_rdd, WANTED_VEHICLE, channels, MIN_MSG_PER_SEGMENT, MARKER)

        data_segment_rdd = (
            # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msgs)

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
            .flatMapValues(feature_extraction_utils.gen_segment)
            .cache())
        
        segment_count = data_segment_rdd.count()
        glog.info("Get {} segments: [{}, ...]".format(
            segment_count,
            data_segment_rdd.keys().first() if segment_count > 0 else 'None'))

        result = (
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
            .map(file_utils.touch)
            # Trigger actions.
            .count())

        glog.info('Finished %d h5_result_rdd!' % result)


if __name__ == '__main__':
    SampleSetFeatureExtraction().main()
