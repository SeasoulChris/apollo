#!/usr/bin/env python
"""
This is a module to extraction features from records
with folder path as part of the key
"""
import os

import h5py
import numpy as np
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.features.features import GetDatapoints
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.common_feature_extraction as CommonFE


class GeneralFeatureExtractionPipeline(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'general_feature_extraction')

    def run_test(self):
        """Run test."""
        records = [
            '/apollo/modules/data/fuel/testdata/control/left_40_10/1.record.00000',
            '/apollo/modules/data/fuel/testdata/control/transit/1.record.00000',
        ]
        origin_prefix = '/apollo/modules/data/fuel/testdata/control'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated'

        dir_to_records = self.get_spark_context().parallelize(
            records).keyBy(os.path.dirname)
        self.run(dir_to_records, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/2019/'
        target_prefix = 'modules/control/feature_extraction_hf5/2019/'

        files = s3_utils.list_files(bucket, origin_prefix).cache()
        complete_dirs = files.filter(
            lambda path: path.endswith('/COMPLETE')).map(os.path.dirname)
        dir_to_records = files.filter(
            record_utils.is_record_file).keyBy(os.path.dirname)
        self.run(spark_op.filter_keys(dir_to_records, complete_dirs),
                 origin_prefix, target_prefix)

    def run(self, dir_to_records_rdd, origin_prefix, target_prefix):
        """ processing RDD """
        wanted_chs = ['/apollo/canbus/chassis',
                      '/apollo/localization/pose']
        max_phase_delta = 0.01
        min_segment_length = 10

        def build_training_dataset(chassis, pose):
            """align chassis and pose data and build data segment"""
            chassis.sort(key=lambda x: x.header.timestamp_sec)
            pose.sort(key=lambda x: x.header.timestamp_sec)
            # In the record, control and chassis always have same number of frames
            times_pose = np.array([x.header.timestamp_sec for x in pose])
            times_cs = np.array([x.header.timestamp_sec for x in chassis])

            index = [0, 0]

            def align():
                """align up chassis and pose data w.r.t time """
                while index[0] < len(times_cs) and index[1] < len(times_pose) \
                        and abs(times_cs[index[0]] - times_pose[index[1]]) > max_phase_delta:
                    while index[0] < len(times_cs) and index[1] < len(times_pose) \
                            and times_cs[index[0]] < times_pose[index[1]] - max_phase_delta:
                        index[0] += 1
                    while index[0] < len(times_cs) and index[1] < len(times_pose) \
                            and times_pose[index[1]] < times_cs[index[0]] - max_phase_delta:
                        index[1] += 1

            align()

            while index[0] < len(times_cs)-1 and index[1] < len(times_pose)-1:
                limit = min(len(times_cs)-index[0], len(times_pose)-index[1])

                for seg_len in range(1, limit):
                    delta = abs(times_cs[index[0]+seg_len]
                                - times_pose[index[1]+seg_len])
                    if delta > max_phase_delta or seg_len == limit-1:
                        if seg_len >= min_segment_length or seg_len == limit - 1:
                            yield GetDatapoints(pose[index[1]:index[1]+seg_len],
                                                chassis[index[0]:index[0]+seg_len])
                            index[0] += seg_len
                            index[1] += seg_len
                            align()
                            break

        def gen_hdf5(elem):
            """ write data segment to hdf5 file """
            folder_path = str(elem[0][0])
            time_stamp = str(elem[0][1])
            out_file_path = "{}/training_dataset_{}.hdf5".format(
                folder_path.replace(origin_prefix, target_prefix, 1),
                time_stamp)
            out_dir = os.path.dirname(out_file_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            out_file = h5py.File(out_file_path, "w")
            chassis = elem[1][0]
            pose = elem[1][1]
            i = 0
            for mini_dataset in build_training_dataset(chassis, pose):
                name = "_segment_" + str(i).zfill(3)
                out_file.create_dataset(
                    name, data=mini_dataset, dtype="float32")
                i += 1
            out_file.close()
            return elem
        wanted_vehicle = 'Transit'

        folder_vehicle_rdd = (dir_to_records_rdd
                              .flatMapValues(record_utils.read_record(['/apollo/hmi/status']))
                              # parse message
                              .mapValues(record_utils.message_to_proto)
                              .mapValues(lambda elem: elem.current_vehicle)
                              .filter(lambda elem: elem[1] == wanted_vehicle)
                              # remove duplication of folders
                              .distinct()
                              # choose only folder path
                              .map(lambda x: x[0]))
        print folder_vehicle_rdd.count()
        # print folder_vehicle_rdd.take(1)

        channels_rdd = (folder_vehicle_rdd
                        .keyBy(lambda x: x)
                        # record path
                        .flatMapValues(CommonFE.folder_to_record)
                        # read message
                        .flatMapValues(record_utils.read_record(wanted_chs))
                        # parse message
                        .mapValues(record_utils.message_to_proto))
        print channels_rdd.count()
        # print channels_rdd.take(1)

        pre_segment_rdd = (channels_rdd
                           # choose time as key, group msg into 1 sec
                           .map(CommonFE.gen_key)
                           # combine chassis message and pose message with the same key
                           .combineByKey(CommonFE.to_list, CommonFE.append, CommonFE.extend))
        print pre_segment_rdd.count()
        print pre_segment_rdd.take(1)

        data_rdd = (pre_segment_rdd
                    # msg list(path_key,(chassis,pose))
                    .mapValues(CommonFE.process_seg)
                    # align msg, generate data segment, write to hdf5 file.
                    .map(gen_hdf5))

        print data_rdd.count()
        # print data_rdd.take(1)


if __name__ == '__main__':
    GeneralFeatureExtractionPipeline().run_test()
