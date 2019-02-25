#!/usr/bin/env python
""" extracting sample set for mkz7 """
# pylint: disable = fixme
# pylint: disable = no-member
from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.control.features.common_feature_extraction as CommonFE


class SampleSetFeatureExtractionPipeline(BasePipeline):
    """ Generate general feature extraction hdf5 files from records """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'sample_set_feature_extraction')

    def run_test(self):
        """Run test."""
        folder_path = ["/apollo/modules/data/fuel/testdata/control/left_40_10",
                       "/apollo/modules/data/fuel/testdata/control/transit"]

        spark_context = self.get_spark_context()
        records_rdd = spark_context.parallelize(folder_path)

        self.run(records_rdd)

    @staticmethod
    def run(records_rdd):
        """ processing RDD """
        wanted_vehicle = 'Mkz7'
        wanted_chs = ['/apollo/canbus/chassis',
                      '/apollo/localization/pose']

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
        # print(folder_vehicle_rdd.collect())

        channels_rdd = (folder_vehicle_rdd
                        .keyBy(lambda x: x)
                        # record path
                        .flatMapValues(CommonFE.folder_to_record)
                        # read message
                        .flatMapValues(record_utils.read_record(wanted_chs))
                        # parse message
                        .mapValues(record_utils.message_to_proto))

        pre_segment_rdd = (channels_rdd
                           # choose time as key, group msg into 1 sec
                           .map(CommonFE.gen_key)
                           # combine chassis message and pose message with the same key
                           .combineByKey(CommonFE.to_list, CommonFE.append, CommonFE.extend))

        data_rdd = (pre_segment_rdd
                    # msg list
                    .mapValues(CommonFE.process_seg)
                    # flat value to paired data points
                    .flatMapValues(CommonFE.pair_cs_pose)
                    .map(CommonFE.get_data_point))

        # data feature set
        featured_data_rdd = (data_rdd
                             .map(CommonFE.feature_key_value)
                             .combineByKey(CommonFE.to_list, CommonFE.append, CommonFE.extend))

        data_segment_rdd = (featured_data_rdd
                            # generate segment w.r.t time
                            .mapValues(CommonFE.gen_segment)
                            # write all segment into a hdf5 file
                            .map(CommonFE.write_h5_with_key))
        print data_segment_rdd.count()


if __name__ == '__main__':
    SampleSetFeatureExtractionPipeline().run_test()
