#!/usr/bin/env python
"""Extraction features from records with folder path as part of the key"""
from collections import Counter
import glob
import operator
import os

import colored_glog as glog
import h5py
import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.features import GetDatapoints
import fueling.common.bos_client as bos_client
import fueling.common.file_utils as file_utils
import fueling.common.h5_utils as h5_utils
import fueling.common.record_utils as record_utils
import fueling.control.dynamic_model.conf.model_config as model_config
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_rdd_utils as feature_extraction_rdd_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils


channels = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}
VEHICLE = feature_extraction_utils.WANTED_VEHICLE
MIN_MSG_PER_SEGMENT = 10
MAX_PHASE_DELTA_SEGMENT = 0.015
MARKER = 'CompleteGoldenSet'


def write_h5(elem, folder_path, file_name):
    """write to h5 file, use feature key as file name"""
    file_utils.makedirs(folder_path)
    with h5py.File("{}/{}.hdf5".format(folder_path, file_name), "w") as out_file:
        i = 0
        for data_set in elem:
            name = "_segment_" + str(i).zfill(3)
            out_file.create_dataset(name, data=data_set, dtype="float32")
            i += 1


def write_segment(elem, origin_prefix, target_prefix):
    """write to h5 file, use feature key as file name"""
    folder_path, data_set = elem
    folder_path = folder_path.replace(origin_prefix, target_prefix, 1)
    file_name = '1'
    write_h5(data_set, folder_path, file_name)
    return folder_path


def gen_segment(elem):
    """ generate segment w.r.t time """
    segments = []
    pre_time = elem[0][0]
    data_set = np.array(elem[0][1])
    counter = 1  # count segment length first element
    for i in range(1, len(elem)):
        if (elem[i][0] - pre_time) <= MAX_PHASE_DELTA_SEGMENT:
            data_set = np.vstack([data_set, elem[i][1]])
            counter += 1
        else:
            if counter > model_config.feature_config['sequence_length']:
                segments.append(data_set)
            data_set = np.array([elem[i][1]])
            counter = 0
        pre_time = elem[i][0]
    if counter > model_config.feature_config['sequence_length']:
        segments.append(data_set)
    return segments


def get_data_point(elem):
    """ extract data from msg """
    chassis, pose_pre = elem
    pose = pose_pre.pose
    res = np.array([
        pose.heading,  # 0
        pose.orientation.qx,  # 1
        pose.orientation.qy,  # 2
        pose.orientation.qz,  # 3
        pose.orientation.qw,  # 4
        pose.linear_velocity.x,  # 5
        pose.linear_velocity.y,  # 6
        pose.linear_velocity.z,  # 7
        pose.linear_acceleration.x,  # 8
        pose.linear_acceleration.y,  # 9
        pose.linear_acceleration.z,  # 10
        pose.angular_velocity.x,  # 11
        pose.angular_velocity.y,  # 12
        pose.angular_velocity.z,  # 13
        chassis.speed_mps,  # 14 speed
        chassis.throttle_percentage / 100,  # 15 throttle
        chassis.brake_percentage / 100,  # 16 brake
        chassis.steering_percentage / 100,  # 17
        chassis.driving_mode,  # 18
        pose.position.x,  # 19
        pose.position.y,  # 20
        pose.position.z,  # 21
    ])
    return (chassis.header.timestamp_sec, res)


class GoldenSet(BasePipeline):
    """ Generate sample set feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'golden_set_feature_extraction')

    def run_test(self):
        """Run test."""
        glog.info('VEHICLE: %s' % VEHICLE)

        origin_dir = '/apollo/data/GoldenSet'
        target_dir = os.path.join('/apollo/modules/data/fuel/testdata/control/generated',
                                  'GoldenSet', VEHICLE)
        # RDD(record_dirs)
        todo_tasks = self.to_rdd([origin_dir])

        todo_records = spark_helper.cache_and_log(
            'todo_records',
            todo_tasks
            # RDD(record_file)
            .flatMap(lambda path: glob.glob(os.path.join(path, '*/*/*recover')))
            # PairRDD(dir, record_file)
            .keyBy(os.path.dirname))

        glog.info('todo_recors: %s' % todo_records.collect())
        self.run(todo_records, origin_dir, target_dir)

    def run_prod(self):
        """Run prod."""
        origin_prefix = 'modules/control/data/records/Mkz7/2019-04-30'
        target_prefix = os.path.join(
            'modules/control/data/results/GoldenSet', VEHICLE, '2019-04-30')

        todo_records = spark_helper.cache_and_log(
            'todo_records',
            # RDD(record_file)
            self.to_rdd(self.bos().list_files(origin_prefix, '.recover'))
            # PairRDD(dir, record_file)
            .keyBy(os.path.dirname))

        target_dir = bos_client.abs_path(target_prefix)
        origin_dir = bos_client.abs_path(origin_prefix)

        self.run(todo_records, origin_dir, target_dir)

    def run(self, todo_records, origin_dir, target_dir):
        # wanted vehicle is know and the folder only include wanted vehicle
        # RDD(aboslute_dir) which include records of the wanted vehicle
        dir_to_msgs = spark_helper.cache_and_log(
            'dir_to_msgs',
            todo_records
            # PairRDD(dir, msgs)
            .flatMapValues(record_utils.read_record(channels)))

        valid_msgs = spark_helper.cache_and_log(
            'valid_msgs',
            dir_to_msgs
            # PairRDD(dir, topic_counter)
            .mapValues(lambda msg: Counter([msg.topic]))
            # PairRDD(dir, topic_counter)
            .reduceByKey(operator.add)
            # PairRDD(dir, topic_counter)
            .filter(lambda (_, counter):
                    counter.get(record_utils.CHASSIS_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT
                    and counter.get(record_utils.LOCALIZATION_CHANNEL, 0) >= MIN_MSG_PER_SEGMENT)
            # RDD(dir)
            .keys())

        # PairRDD(dir, valid_msgs)
        valid_msgs = spark_op.filter_keys(dir_to_msgs, valid_msgs)

        data_segment_rdd = spark_helper.cache_and_log(
            'parsed_msg',
            # PairRDD(dir_segment, (chassis_msg_list, pose_msg_list))
            feature_extraction_rdd_utils.chassis_localization_parsed_msg_rdd(valid_msgs))

        data_segment_rdd = spark_helper.cache_and_log(
            'pair_cs_pose',
            data_segment_rdd
            # PairRDD(dir_segment, paired_chassis_msg_pose_msg)
            .flatMapValues(feature_extraction_utils.pair_cs_pose))

        data_segment_rdd = spark_helper.cache_and_log(
            'get_data_point',
            data_segment_rdd
            # PairRDD(dir, data_point)
            .mapValues(get_data_point))

        data_segment_rdd = spark_helper.cache_and_log(
            'data_segment',
            data_segment_rdd
            .groupByKey()
            # PairRDD(dir, list of (timestamp_sec, data_point))
            .mapValues(list))

        data_segment_rdd = spark_helper.cache_and_log(
            'data_segment',
            # PairRDD(dir, (timestamp_sec, data_point))
            data_segment_rdd.mapValues(gen_segment))

        spark_helper.cache_and_log(
            'hdf5',
            data_segment_rdd.map(lambda elem: write_segment(elem, origin_dir, target_dir)))


if __name__ == '__main__':
    GoldenSet().main()
