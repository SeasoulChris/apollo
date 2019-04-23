#!/usr/bin/env python
import operator
import os

import colored_glog as glog

from fueling.common.base_pipeline import BasePipeline
import fueling.common.db_backed_utils as db_backed_utils
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils


class DataForLearning(BasePipeline):
    """Records to DataForLearning proto pipeline."""
    def __init__(self):
        BasePipeline.__init__(self, 'data-for-learning')

    def run_test(self):
        """Run test."""
        # RDD(dir_path)
        records_dir = self.context().parallelize(['/apollo/docs/demo_guide'])
        origin_prefix = '/apollo/docs/demo_guide'
        target_prefix = '/apollo/data/prediction/features'
        self.run(records_dir, origin_prefix, target_prefix)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        origin_prefix = 'small-records'
        target_prefix = 'modules/prediction/features'

        records_dir = (
            # RDD(file), start with origin_prefix
            s3_utils.list_files(bucket, origin_prefix)
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct())
        self.run(records_dir, origin_prefix, target_prefix)

    def run(self, record_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(record_dir)
            record_dir_rdd
            # PairRDD(record_dir, map_name)
            .mapPartitions(self.get_dirs_map)
            # RDD(0/1), 1 for success
            # .map(lambda dir_map: self.process_dir(
            #     dir_map[0],
            #     record_dir.replace(origin_prefix, os.path.join(target_prefix, dir_map[1]), 1)),
            #     dir_map[1])
            .cache())
        print(result.collect())
        # glog.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    @staticmethod
    def process_dir(record_dir, target_dir, map_name):
        """Call prediction C++ code."""
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/prediction/data_pipelines/scripts/records_to_data_for_learning.sh '
            '"{}" "{}" "{}"'.format(record_dir, target_dir, map_name))
        print(command)
        return 1
        if os.system(command) == 0:
            glog.info('Successfuly processed {} to {}'.format(record_dir, target_dir))
            return 1
        else:
            glog.error('Failed to process {} to {}'.format(record_dir, target_dir))
        return 0

    def get_dirs_map(self, record_dirs):
        """Return the (record_dir, map_name) pair"""
        record_dirs = list(record_dirs)
        collection = self.mongo().record_collection()
        dir_map_dict = db_backed_utils.lookup_map_for_dirs(record_dirs, collection)
        dir_map_list = []
        for record_dir in record_dirs:
            map_name = dir_map_dict.get(record_dir)
            if map_name is None:
                continue
            if map_name.find("Sunnyvale") >= 0:
                dir_map_list.append((record_dir, "sunnyvale_with_two_offices"))
            if map_name.find("San Mateo") >= 0:
                dir_map_list.append((record_dir, "san_mateo"))
        return dir_map_list


if __name__ == '__main__':
    DataForLearning().main()
