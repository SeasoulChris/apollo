#!/usr/bin/env python
"""
This is a module to extraction features from records
with folder path as part of the key
"""
import h5py
import numpy as np

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
from fueling.control.features.features import GetDatapoints
import fueling.control.features.common_feature_extraction as CommonFE


class GeneralFeatureExtractionPipeline(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'general_feature_extraction')

    def run_test(self):
        """Run test."""
        folder_path = ["/apollo/modules/data/fuel/testdata/modules/control/left_40_10",
                       "/apollo/modules/data/fuel/testdata/modules/control/right_40_10"]

        spark_context = self.get_spark_context()
        records_rdd = spark_context.parallelize(folder_path)

        self.run(records_rdd)

    @staticmethod
    def run(records_rdd):
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
            out_file = h5py.File(
                "{}/training_dataset_{}.hdf5".format(folder_path, time_stamp), "w")

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

        channels_rdd = (records_rdd
                        # add foler path as key
                        .keyBy(lambda x: x)
                        # record path
                        .flatMapValues(CommonFE.folder_to_record)
                        # read message
                        .flatMapValues(record_utils.read_record(wanted_chs))
                        # parse message
                        .mapValues(CommonFE.process_msg))

        pre_segment_rdd = (channels_rdd
                           # choose time as key, group msg into 1 sec
                           .map(CommonFE.gen_key)
                           # combine chassis message and pose message with the same key
                           .combineByKey(CommonFE.to_list, CommonFE.append, CommonFE.extend))

        data_rdd = (pre_segment_rdd
                    # msg list(path_key,(chassis,pose))
                    .mapValues(CommonFE.process_seg)
                    # align msg, generate data segment, write to hdf5 file.
                    .map(gen_hdf5))
        data_rdd.count()


if __name__ == '__main__':
    GeneralFeatureExtractionPipeline().run_test()
