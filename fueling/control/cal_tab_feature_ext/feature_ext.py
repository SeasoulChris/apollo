#!/usr/bin/env python
import fueling.common.record_utils as record_utils
import fueling.control.features.common_cal_tab as commonCalTab
import fueling.control.features.common_feature_extraction as CommonFE
from fueling.common.base_pipeline import BasePipeline
"""
This is a module to extraction features from records
with folder path as part of the key
"""
import os
from scipy.misc import imsave
import numpy as np
import pyspark_utils.op as spark_op


class CalTabFeatureExtPipeline(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'feature_ext')

    def run_test(self):
        """Run test."""
        folder_path = ["/apollo/modules/data/fuel/testdata/control/transient_1",
                       '/apollo/modules/data/fuel/testdata/control/mkz7_s/']

        spark_context = self.get_spark_context()
        records_rdd = spark_context.parallelize(folder_path)

        self.run(records_rdd)

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
                              .flatMapValues(CommonFE.folder_to_record)
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
                        .flatMapValues(CommonFE.folder_to_record)
                        # read message
                        .flatMapValues(record_utils.read_record(wanted_chs))
                        # parse message
                        .mapValues(record_utils.message_to_proto))
        print channels_rdd.count()

        # aggregate msg to segment
        pre_segment_rdd = (channels_rdd
                           # choose time as key, group msg into 1 sec
                           .map(CommonFE.gen_key)
                           # combine chassis message and pose message with the same key
                           .combineByKey(CommonFE.to_list, CommonFE.append, CommonFE.extend)
                           # msg list(path_key,(chassis,pose))
                           .mapValues(CommonFE.process_seg))

        print pre_segment_rdd.count()

        data_rdd = (pre_segment_rdd
                    # ((folder,time/min),(chassis,pose))
                    .mapValues(CommonFE.pair_cs_pose)
                    # ((folder,time/min),feature_matrix)
                    # feature generator
                    .mapValues(commonCalTab.feature_generate)
                    # process feature: feature filter
                    .mapValues(commonCalTab.feature_filter)
                    # process feature: feature cut
                    .mapValues(commonCalTab.feature_cut)
                    # process feature: feature distribute
                    .mapValues(commonCalTab.feature_distribute)
                    # process feature: feature store
                    .mapValues(commonCalTab.feature_store)
                    # write features to hdf5 files
                    .map(commonCalTab.write_h5_train_test))

        print data_rdd.count()


if __name__ == '__main__':
    CalTabFeatureExtPipeline().run_test()
