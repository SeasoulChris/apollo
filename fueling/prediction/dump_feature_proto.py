#!/usr/bin/env python
import operator
import os

from fueling.common.base_pipeline import BasePipeline
import fueling.common.context_utils as context_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

SKIP_EXISTING_DST_FILE = False


class DumpFeatureProto(BasePipeline):
    """Records to feature proto pipeline."""
    def __init__(self, origin_prefix, target_prefix):
        super(DumpFeatureProto, self).__init__()
        self.origin_prefix = origin_prefix
        self.target_prefix = target_prefix

    def run(self):
        """Run prod."""
        records_dir = (
            # RDD(file), start with origin_prefix
            self.to_rdd(self.our_storage().list_files(self.origin_prefix))
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct())

        completed_records_dir = (
            # RDD(label_dir). start with target_prefix
            self.to_rdd(self.our_storage().list_end_dirs(self.target_prefix))
            # RDD(record_dir), has been completed
            .map(lambda label_dir: label_dir.replace(os.path.join(
                self.target_prefix, label_dir[(label_dir.find(self.target_prefix)
                                              + len(self.target_prefix)):].split('/')[0] + '/'),
                self.origin_prefix))
            # RDD(record_dir), which is unique
            .distinct())
        # RDD(todo_records_dir)
        todo_records_dir = records_dir

        if SKIP_EXISTING_DST_FILE:
            # RDD(todo_records_dir)
            todo_records_dir = todo_records_dir.subtract(completed_records_dir).distinct()

        self.run_internal(todo_records_dir, self.origin_prefix, self.target_prefix)

    def run_internal(self, records_dir_rdd, origin_prefix, target_prefix):
        """Run the pipeline with given arguments."""
        result = (
            # RDD(record_dir)
            records_dir_rdd
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

    @staticmethod
    def process_dir(record_dir, target_dir, map_name):
        """Call prediction C++ code."""
        additional_ld_path = "/usr/local/miniconda/envs/fuel/lib/"
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/prediction/data_pipelines/scripts/records_to_dump_feature_proto.sh '
            '"{}" "{}" "{}" "{}"'.format(record_dir, target_dir, map_name, additional_ld_path))
        if os.system(command) == 0:
            logging.info('Successfully processed {} to {}'.format(record_dir, target_dir))
            return 1
        else:
            logging.error('Failed to process {} to {}'.format(record_dir, target_dir))
        return 0

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
        if "houston" in path_lower:
            return "houston"
        return "demo"

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
    origin_prefix = 'modules/prediction/kinglong/'
    target_prefix = 'modules/prediction/kinglong_labels/'
    if context_utils.is_local():
        origin_prefix = "/fuel/kinglong_data/records/"
        target_prefix = "/fuel/kinglong_data/labels/"
    if context_utils.is_local():
        DumpFeatureProto(origin_prefix, target_prefix).main()
