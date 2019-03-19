#!/usr/bin/env python
"""Extraction features from records with folder path as part of the key"""

from collections import Counter
import operator
import os

import h5py
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.features import GetDatapoints
import fueling.common.colored_glog as glog
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.common.time_utils as time_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

WANTED_VEHICLE = feature_extraction_utils.WANTED_VEHICLE
MIN_MSG_PER_SEGMENT = 100


class GeneralFeatureExtraction(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'general_feature_extraction')

    def run_test(self):
        """Run test."""
        records = [
            'modules/data/fuel/testdata/control/left_40_10/1.record.00000',
            'modules/data/fuel/testdata/control/transit/1.record.00000',
        ]

        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = 'modules/data/fuel/testdata/control/generated'
        root_dir = '/apollo'
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
        dir_to_records = files.filter(
            record_utils.is_record_file).keyBy(os.path.dirname)
        root_dir = s3_utils.S3_MOUNT_PATH
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix, root_dir)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix, root_dir):
        """ processing RDD """
        def _gen_hdf5(elem):
            """ write data segment to hdf5 file """
            # glog.info("Processing data in folder:" % str(elem[0][0]))
            (folder_path, segment_id), (chassis, pose) = elem
            glog.info("Processing data in folder: %s" % folder_path)
            out_dir = folder_path.replace(origin_prefix, target_prefix, 1)
            file_utils.makedirs(out_dir)
            out_file_path = "{}/{}_{}.hdf5".format(
                out_dir, WANTED_VEHICLE, segment_id)
            with h5py.File(out_file_path, "w") as out_file:
                i = 0
                for mini_dataset in self.build_training_dataset(chassis, pose):
                    name = "_segment_" + str(i).zfill(3)
                    out_file.create_dataset(
                        name, data=mini_dataset, dtype="float32")
                    i += 1
            glog.info("Created all mini_dataset to {}".format(out_file_path))
            return elem

        # -> (dir, record), in absolute path
        dir_to_records = dir_to_records_rdd.map(lambda x: (os.path.join(root_dir, x[0]),
                                                           os.path.join(root_dir, x[1]))).cache()

        selected_vehicles = (
            # -> (dir, vehicle)
            self.get_vehicle_of_dirs(dir_to_records)
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
            .map(self.gen_segment)
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

        result = (
            # -> (dir_segment, msg)
            spark_op.filter_keys(dir_to_msgs, valid_segments)
            # -> (dir_segment, msgs)
            .groupByKey()
            # -> (dir_segment, proto_dict)
            .mapValues(record_utils.messages_to_proto_dict())
            # -> (dir_segment, (chassis_list, pose_list))
            .mapValues(lambda proto_dict: (proto_dict[record_utils.CHASSIS_CHANNEL],
                                           proto_dict[record_utils.LOCALIZATION_CHANNEL]))
            .map(_gen_hdf5))
        glog.info('Generated %d h5 files!' % result.count())

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
    def gen_segment(dir_to_msg):
        """Generate new key which contains a segment id part."""
        task_dir, msg = dir_to_msg
        dt = time_utils.msg_time_to_datetime(msg.timestamp)
        segment_id = dt.strftime('%Y%m%d-%H%M')
        return ((task_dir, segment_id), msg)

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

    GeneralFeatureExtraction().run_test()
