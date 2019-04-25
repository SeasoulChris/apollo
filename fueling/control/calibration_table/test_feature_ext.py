#!/usr/bin/env python
import fueling.common.record_utils as record_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils
import fueling.control.features.calibration_table_utils as calibration_table_utils
from fueling.common.base_pipeline import BasePipeline
"""
This is a module to extraction features from records
with folder path as part of the key
"""
import os
import numpy as np
import pyspark_utils.op as spark_op


class CalTabFeatureExtPipeline(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'feature_ext')

    def run_test(self):
        """Run test."""
        folder_path = ["/apollo/modules/data/fuel/testdata/control/transient_1"]
        self.run(self.to_rdd(folder_path))

    @staticmethod
    def run(records_rdd):
        """ processing RDD """

        # parameters
        wanted_vehicle = 'Mkz7'
        # wanted_vehicle = 'Transit'
        wanted_chs = ['/apollo/canbus/chassis',
                      '/apollo/localization/pose']

        # choose record for wanted vehicle type
        folder_vehicle_rdd = (records_rdd
                              # key as folder path
                              .keyBy(lambda x: x)
                              .flatMapValues(feature_extraction_utils.folder_to_record)
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

        # record to msg, (folder_dir,msg)
        channels_rdd = (folder_vehicle_rdd
                        .keyBy(lambda x: x)
                        # record path
                        .flatMapValues(feature_extraction_utils.folder_to_record)
                        # read message
                        .flatMapValues(record_utils.read_record(wanted_chs))
                        # parse message
                        .mapValues(record_utils.message_to_proto))
        print channels_rdd.count()

        # aggregate msg to segment
        pre_segment_rdd = (channels_rdd
                           # choose time as key, group msg into 1 sec
                           .map(feature_extraction_utils.gen_key)
                           # combine chassis message and pose message with the same key
                           .combineByKey(feature_extraction_utils.to_list, feature_extraction_utils.append, feature_extraction_utils.extend)
                           # msg list(path_key,(chassis,pose))
                           .mapValues(feature_extraction_utils.process_seg))

        print pre_segment_rdd.count()

        data_rdd = (pre_segment_rdd
                    # ((folder,time/min),(chassis,pose))
                    .mapValues(feature_extraction_utils.pair_cs_pose)
                    # ((folder,time/min),feature_matrix)
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
                    .map(calibration_table_utils.write_h5_train_test))

        print data_rdd.count()


if __name__ == '__main__':
    CalTabFeatureExtPipeline().main()
