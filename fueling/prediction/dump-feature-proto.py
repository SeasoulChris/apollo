#!/usr/bin/env python
import glob
import operator
import os

import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.colored_glog as glog
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class DumpFeatureProto(BasePipeline):
    """Records to feature proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'dump-feature-proto-san-mateo')

    def run_test(self):
        """Run test."""
        sc = self.get_spark_context()
        root_dir = '/apollo'
        records_dir = sc.parallelize(['docs/demo_guide'])
        origin_prefix = 'docs/demo_guide'
        target_prefix = 'data/prediction/labels-san-mateo'
        self.run(root_dir, records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        root_dir = s3_utils.S3_MOUNT_PATH
        bucket = 'apollo-platform'
        origin_prefix = 'small-records/'
        target_prefix = 'modules/prediction/labels-san-mateo/'

        records_dir = (
            # file, start with origin_prefix
            s3_utils.list_files(bucket, origin_prefix)
            # -> record_file
            .filter(record_utils.is_record_file)
            # -> record_dir
            .map(os.path.dirname)
            # -> record_dir, which is unique
            .distinct())
        self.run(root_dir, records_dir, origin_prefix, target_prefix)

    def run(self, root_dir, records_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # record_dir
            records_dir_rdd
            # -> (record_dir, target_dir)
            .map(lambda record_dir: (record_dir,
                                     record_dir.replace(origin_prefix, target_prefix, 1)))
            # -> (record_dir, target_dir), in absolute path
            .map(lambda src_dst: (os.path.join(root_dir, src_dst[0]),
                                  os.path.join(root_dir, src_dst[1])))
            # -> 0/1
            .map(spark_op.do_tuple(self.process_dir))
            .cache())
        glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(src_dir, target_dir):
        """Call prediction C++ code."""
        # use /apollo/hmi/status's current_map entry to match map info
        map_dir = get_map_name_from_records(src_dir)
        command = (
            'cd /apollo && '
            'bash modules/tools/prediction/data_pipelines/scripts/records_to_dump_feature_proto.sh '
            '"{}" "{}" "{}"'.format(src_dir, target_dir, map_dir))
        if os.system(command) == 0:
            glog.info('Successfuly processed {} to {}'.format(src_dir, target_dir))
            return 1
        else:
            glog.error('Failed to process {} to {}'.format(src_dir, target_dir))
        return 0


def get_map_name_from_records(records_dir):
    """Get the map_name from a records_dir by /apollo/hmi/status channel"""
    map_list = os.listdir('/apollo/modules/map/data/')
    # get the map_dict mapping follow the hmi Titlecase. E.g.: "Hello World" -> "hello_world".
    map_dict = {map_name.replace('_', ' ').title(): map_name for map_name in map_list}
    reader = record_utils.read_record([record_utils.HMI_STATUS_CHANNEL])
    glog.info('Try getting map name from {}'.format(records_dir))
    records = glob.glob(os.path.join(records_dir,'*.record*'))
    for record in records:
        for msg in reader(record):
            hmi_status = record_utils.message_to_proto(msg)
            map_name = map_dict[str(hmi_status.current_map)]
            glog.info('Get map name "{}" from record {}'.format(map_name, record))
            return map_name
    glog.error('Failed to get map_name')

if __name__ == '__main__':
    DumpFeatureProto().run_prod()
