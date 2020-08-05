#!/usr/bin/env python
""" extracting sample set """
import glob
import os

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.common.spark_helper as spark_helper
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
WANTED_VEHICLE = feature_extraction_utils.FEATURE_KEY.vehicle_type
MIN_MSG_PER_SEGMENT = 100
MARKER = 'CompleteBackwardSampleSet'


class BackwardSampleSet(BasePipeline):
    """ Generate sample set feature extraction hdf5 files from records """

    def run_test(self):
        """Run test."""
        logging.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)
        origin_prefix = '/fuel/testdata/control/backward_records'
        target_prefix = os.path.join('/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'BackwardSampleSet')
        # RDD(record_dirs)
        todo_tasks = (self.to_rdd([origin_prefix])
                      .flatMap(lambda path: glob.glob(os.path.join(path, '*'))))
        print(todo_tasks.first())
        # PairRDD(record_dirs, record_files)
        todo_records = spark_helper.cache_and_log('todo_records',
                                                  dir_utils.get_todo_records(todo_tasks))

        self.run_internal(todo_records, origin_prefix, target_prefix)

    def run(self):
        """Run prod."""
        origin_prefix = 'modules/control/data/records/Mkz7'
        target_prefix = os.path.join('modules/control/data/results',
                                     'BackwardSampleSet', WANTED_VEHICLE)
        # RDD(record_dirs)
        """ get to do jobs """
        todo_tasks = dir_utils.get_todo_tasks(origin_prefix, target_prefix, 'COMPLETE', MARKER)
        # PairRDD(record_dirs, record_files)
        todo_records = spark_helper.cache_and_log('todo_records',
                                                  dir_utils.get_todo_records(todo_tasks))
        self.run_internal(todo_records, origin_prefix, target_prefix)

    def run_internal(self, dir_to_records_rdd, origin_prefix, target_prefix):
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
            chassis_localization_segment_rdd(dir_to_msgs, MIN_MSG_PER_SEGMENT), 0)

        # PairRDD((dir_segment, segment_id), msg)
        valid_msgs = spark_helper.cache_and_log(
            'ValidMsg',
            feature_extraction_rdd_utils. valid_msg_rdd(dir_to_msgs, valid_segments), 0)

        data_segment_rdd = spark_helper.cache_and_log(
            'parsed_msg',
            # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(
                valid_msgs), 0)

        data_segment_rdd = spark_helper.cache_and_log(
            'pair_cs_pose',
            data_segment_rdd
            # PairRDD((dir_segment, segment_id), paired_chassis_msg_pose_msg)
            .flatMapValues(feature_extraction_utils.pair_cs_pose), 0)

        data_segment_rdd = spark_helper.cache_and_log(
            'get_data_point',
            data_segment_rdd
            # PairRDD((dir, timestamp_sec), signle data_point)
            .map(feature_extraction_utils.get_data_point), 0)

        # same size
        data_segment_rdd = spark_helper.cache_and_log(
            'feature_key_value',
            data_segment_rdd
            # PairRDD((dir, feature_key), (timestamp_sec, data_point))
            .map(feature_extraction_utils.gen_feature_key_backwards), 0)

        logging.info('number of elems: %d' % data_segment_rdd
                     # PairRDD((dir, feature_key), (timestamp_sec, data_point) RDD)
                     .groupByKey()
                     # PairRDD((dir, feature_key), list of (timestamp_sec, data_point))
                     .mapValues(list).count())

        data_segment_rdd = spark_helper.cache_and_log(
            'gen_segment',
            data_segment_rdd
            # remove the forward data
            .filter(lambda _feature_key_: _feature_key_[0][1] != 10000)
            # PairRDD((dir, feature_key), (timestamp_sec, data_point)s)
            .groupByKey()
            # PairRDD((dir, feature_key), list of (timestamp_sec, data_point))
            .mapValues(list)
            # # PairRDD((dir, feature_key), one segment)
            .flatMapValues(feature_extraction_utils.gen_segment), 0)

        # logging.info('ALL segment: %s' % data_segment_rdd
        #           .map(lambda (key, (time_stamp, segment))
        #                :('Mkz7', segment.shape[0])).reduceByKey(operator.add).collect())

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
    BackwardSampleSet().main()
