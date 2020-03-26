#!/usr/bin/env python
import operator
import os

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.common.mongo_utils import Mongo
import fueling.common.db_backed_utils as db_backed_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


class DataForLearning(BasePipeline):
    """Records to DataForLearning proto pipeline."""
    def run(self):
        """Run prod."""
        origin_prefix = "/fuel/kinglong_data/records/"
        target_prefix = "/fuel/kinglong_data/data_for_learning/"
        if flags.FLAGS.cloud:
            origin_prefix = 'modules/prediction/kinglong/'
            target_prefix = 'modules/prediction/kinglong_data_for_learning/'

        records_dir = (
            # RDD(file), start with origin_prefix
            self.to_rdd(self.our_storage().list_files(origin_prefix))
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct())
        self.run_internal(records_dir, origin_prefix, target_prefix)

    def run_internal(self, record_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(record_dir)
            record_dir_rdd
            # PairRDD(record_dir, map_name)
            .mapPartitions(self.get_dirs_map)
            # RDD(0/1), 1 for success
            .map(lambda dir_map: self.process_dir(
                dir_map[0],
                dir_map[0].replace(origin_prefix,
                                   os.path.join(target_prefix, dir_map[1] + '/'), 1),
                dir_map[1]))
            .cache())
        if result.isEmpty():
            logging.info("Nothing to be processed, everything is under control!")
            return
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    def get_map_dir_by_path(self, path):
        path_lower = path.lower()
        if "baidudasha" in path_lower:
            return "baidudasha"
        if "xionganshiminzhongxin" in path_lower:
            return "XiongAn"
        if "xiamen" in path_lower:
            return "XiaMen"
        if "feifengshan" in path_lower:
            return "FuZhouFeiFengShan"
        return "sunnyvale"

    @staticmethod
    def process_dir(record_dir, target_dir, map_name):
        """Call prediction C++ code."""
        additional_ld_path = "/usr/local/miniconda/envs/fuel/lib/"
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/prediction/data_pipelines/scripts/records_to_data_for_learning.sh '
            '"{}" "{}" "{}" "{}"'.format(record_dir, target_dir, map_name, additional_ld_path))
        if os.system(command) == 0:
            logging.info('Successfully processed {} to {}'.format(record_dir, target_dir))
            return 1
        else:
            logging.error('Failed to process {} to {}'.format(record_dir, target_dir))
        return 0

    def get_dirs_map(self, record_dirs):
        """Return the (record_dir, map_name) pair"""
        dir_map_list = []
        record_dirs = list(record_dirs)

        """ For US data
        collection = Mongo().record_collection()
        dir_map_dict = db_backed_utils.lookup_map_for_dirs(record_dirs, collection)
        for record_dir, map_name in dir_map_dict.items():
            if "Sunnyvale" in map_name:
                dir_map_list.append((record_dir, "sunnyvale"))
            if "San Mateo" in map_name:
                dir_map_list.append((record_dir, "san_mateo"))
        """

        """ For Kinglong data """
        for record_dir in record_dirs:
            map_dir = self.get_map_dir_by_path(record_dir)
            dir_map_list.append((record_dir, map_dir))
        """ End Kinglong data """

        return dir_map_list


if __name__ == '__main__':
    DataForLearning().main()
