#!/usr/bin/env python
""" extracting sample set for mkz7 """
# pylint: disable = fixme
# pylint: disable = no-member
import os

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
        records = ["modules/data/fuel/testdata/control/left_40_10/1.record.00000",
                   "modules/data/fuel/testdata/control/transit/1.record.00000"]

        origin_prefix = 'modules/data/fuel/testdata/control'
        target_prefix = 'modules/data/fuel/testdata/control/generated'
        root_dir = '/apollo'

        dir_to_records = self.get_spark_context().parallelize(
            records).keyBy(os.path.dirname)

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
        wanted_vehicle = 'Mkz7'
        wanted_chs = ['/apollo/canbus/chassis',
                      '/apollo/localization/pose']

        dir_to_records_rdd = dir_to_records_rdd.map(
            lambda x: (os.path.join(
                root_dir, x[0]), os.path.join(root_dir, x[1])))

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
                            .map(lambda elem: CommonFE.write_h5_with_key(elem, origin_prefix, target_prefix, wanted_vehicle)))
        print data_segment_rdd.count()


if __name__ == '__main__':
    SampleSetFeatureExtractionPipeline().run_test()
