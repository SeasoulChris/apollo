#!/usr/bin/env python
import operator
import os

from fueling.common.base_pipeline import BasePipeline
import fueling.common.context_utils as context_utils
from fueling.common.job_utils import JobUtils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


class FrameEnv(BasePipeline):
    """Records to FrameEnv proto pipeline."""
    def __init__(self):
        super(FrameEnv, self).__init__()
        self.is_on_cloud = context_utils.is_cloud()
        self.if_error = False

    def run(self):
        self.input_path = self.FLAGS.get('input_path')
        self.object_storage = self.partner_storage() or self.our_storage()
        self.origin_prefix = os.path.join(self.input_path, 'records')
        self.target_prefix = os.path.join(self.input_path, 'frame_envs')
        records_dir = (
            # RDD(file), start with origin_prefix
            self.to_rdd(self.object_storage.list_files(self.origin_prefix))
            # RDD(record_file)
            .filter(record_utils.is_record_file)
            # RDD(record_dir), with record_file inside
            .map(os.path.dirname)
            # RDD(record_dir), which is unique
            .distinct())
        self.run_internal(records_dir, self.origin_prefix, self.target_prefix)

        if self.is_on_cloud:
            job_id = (self.FLAGS.get('job_id') if self.is_partner_job() else
                      self.FLAGS.get('job_id')[:4])
            JobUtils(job_id).save_job_progress(30)
            if self.if_error:
                error_text = 'Failed to dump frame env proto from record files.'
                JobUtils(job_id).save_job_failure_code('E603')
                JobUtils(job_id).save_job_operations('IDG-apollo@baidu.com',
                                                     error_text, False)

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
        logging.info('Processed {}/{} tasks'.format(result.reduce(operator.add), result.count()))

    def process_dir(self, record_dir, target_dir, map_name):
        """Call prediction C++ code."""
        additional_ld_path = "/usr/local/miniconda/envs/fuel/lib/"
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/prediction/data_pipelines/scripts/records_to_frame_env.sh '
            '"{}" "{}" "{}" "{}"'.format(record_dir, target_dir, map_name, additional_ld_path))
        if os.system(command) == 0:
            logging.info('Successfully processed {} to {}'.format(record_dir, target_dir))
            return 1
        else:
            logging.error('Failed to process {} to {}'.format(record_dir, target_dir))
            self.if_error = True
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
        input_abs_path = self.our_storage().abs_path(self.input_path)
        return self.get_region_from_input_path(input_abs_path)

    def get_region_from_input_path(self, input_path):
        map_path = os.path.join(input_path, 'map/')
        map_list = os.listdir(map_path)
        assert len(map_list) == 1
        map_region_path = os.path.join(map_path, map_list[0])
        index = map_region_path.find('map/')
        if index == -1:
            return ''
        index += 4
        sub_path = map_region_path[index:]
        end = sub_path.find('/')
        if end == -1:
            return sub_path
        return sub_path[:end]

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
    FrameEnv().main()
