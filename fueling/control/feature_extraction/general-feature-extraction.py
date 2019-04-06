#!/usr/bin/env python
"""Extraction features from records with folder path as part of the key"""
import os

import colored_glog as glog
import h5py
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.features import GetDatapoints
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
WANTED_VEHICLE = feature_extraction_utils.WANTED_VEHICLE
MIN_MSG_PER_SEGMENT = 100
MARKER = 'CompleteGeneralSet'

class GeneralFeatureExtraction(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'general_feature_extraction')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)

        origin_prefix = 'modules/data/fuel/testdata/control/sourceData'
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'GeneralSet')
        root_dir = '/apollo'

        list_func = (lambda path: self.get_spark_context().parallelize(
            dir_utils.list_end_files(os.path.join(root_dir, path))))
        # RDD(record_dir)
        todo_tasks = (dir_utils.get_todo_tasks(
            origin_prefix, target_prefix, list_func, '', '/' + MARKER))
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
        self.run(dir_to_records, origin_prefix, target_prefix, root_dir)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = os.path.join('modules/control/feature_extraction_hf5/hdf5_training/',
                                     WANTED_VEHICLE, 'GeneralSet')
        root_dir = s3_utils.S3_MOUNT_PATH
        list_func = (lambda path: s3_utils.list_files(bucket, path))
        # RDD(record_dir)
        todo_tasks_dir = (dir_utils.get_todo_tasks(
            origin_prefix, target_prefix, list_func, '/COMPLETE', '/' + MARKER))

        todo_tasks = (
            # RDD(record_dir)
            todo_tasks_dir
            # RDD(record_files)
            .flatMap(os.listdir)
            # RDD(absolute_record_files)
            .map(lambda record_dir: os.path.join(root_dir, record_dir)))

        # PairRDD(record_dir, record_files)
        dir_to_records = todo_tasks.filter(record_utils.is_record_file).keyBy(os.path.dirname)

        self.run(dir_to_records, origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """
        def _gen_hdf5(elem):
            """ write data segment to hdf5 file """
            # glog.info("Processing data in folder:" % str(elem[0][0]))
            (folder_path, segment_id), (chassis, pose) = elem
            glog.info("Processing data in folder: %s" % folder_path)
            out_dir = folder_path.replace(origin_prefix, target_prefix, 1)
            out_dir = os.path.join(root_dir, out_dir)
            file_utils.makedirs(out_dir)
            out_file_path = "{}/{}.hdf5".format(out_dir, segment_id)
            with h5py.File(out_file_path, "w") as out_file:
                i = 0
                for mini_dataset in self.build_training_dataset(chassis, pose):
                    name = "_segment_" + str(i).zfill(3)
                    out_file.create_dataset(name, data=mini_dataset, dtype="float32")
                    i += 1
            glog.info("Created all mini_dataset to {}".format(out_file_path))
            return elem

        # PairRDD((dir_segment, segment_id), (chassis_msg_list, pose_msg_list))
        valid_msgs = (feature_extraction_rdd_utils
                      .record_to_msgs_rdd(dir_to_records_rdd,
                                          WANTED_VEHICLE, channels, MIN_MSG_PER_SEGMENT, MARKER)
                      .cache())

        result = (
            # PairRDD((dir_segment, segment_id), (chassis_list, pose_list))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msgs)
            # RDD((dir_segment, segment_id), (chassis_list, pose_list))
            # write all segment into a hdf5 file
            .map(_gen_hdf5))
        glog.info('Generated %d h5 files!' % result.count())
        # mark completed folders
        # RDD (dir_segment)
        (feature_extraction_rdd_utils.mark_complete(valid_msgs, origin_prefix,
                                                    target_prefix, MARKER)
         .count())

    @staticmethod
    def get_vehicle_of_dirs(dir_to_records_rdd):
        """
        Extract HMIStatus.current_vehicle from each dir.
        Convert RDD(dir, record) to RDD(dir, vehicle).
        """
        def _get_vehicle_from_records(records):
            reader = record_utils.read_record(
                [record_utils.HMI_STATUS_CHANNEL])
            for record in records:
                glog.info('Try getting vehicle name from {}'.format(record))
                for msg in reader(record):
                    hmi_status = record_utils.message_to_proto(msg)
                    vehicle = hmi_status.current_vehicle
                    glog.info('Get vehicle name "{}" from record {}'.format(
                        vehicle, record))
                    return vehicle
            glog.info('Failed to get vehicle name')
            return ''
        return dir_to_records_rdd.groupByKey().mapValues(_get_vehicle_from_records)

    @staticmethod
    def build_training_dataset(chassis, pose):
        """align chassis and pose data and build data segment"""
        max_phase_delta = 0.01
        min_segment_length = 10
        # In the record, control and chassis always have same number of frames
        times_pose = np.array([x.header.timestamp_sec for x in pose])
        times_cs = np.array([x.header.timestamp_sec for x in chassis])

        glog.info("start time index {} {}".format(times_cs[0], times_pose[0]))
        index = [0, 0]

        def align():
            """align up chassis and pose data w.r.t time """
            while (index[0] < len(times_cs) and index[1] < len(times_pose) and
                   abs(times_cs[index[0]] - times_pose[index[1]]) > max_phase_delta):
                while (index[0] < len(times_cs) and index[1] < len(times_pose) and
                       times_cs[index[0]] < times_pose[index[1]] - max_phase_delta):
                    index[0] += 1
                while (index[0] < len(times_cs) and index[1] < len(times_pose) and
                       times_pose[index[1]] < times_cs[index[0]] - max_phase_delta):
                    index[1] += 1

        align()

        while index[0] < len(times_cs) - 1 and index[1] < len(times_pose) - 1:
            limit = min(len(times_cs) - index[0], len(times_pose) - index[1])

            for seg_len in range(1, limit):
                delta = abs(times_cs[index[0] + seg_len] -
                            times_pose[index[1] + seg_len])
                if delta > max_phase_delta or seg_len == limit - 1:
                    if seg_len >= min_segment_length or seg_len == limit - 1:
                        yield GetDatapoints(pose[index[1]: index[1] + seg_len],
                                            chassis[index[0]: index[0] + seg_len])
                        index[0] += seg_len
                        index[1] += seg_len
                        align()
                        break
        glog.info("build data done")


if __name__ == '__main__':
    GeneralFeatureExtraction().main()
