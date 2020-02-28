#!/usr/bin/env python
import datetime
import os
import time

from fueling.common.base_pipeline_v2 import BasePipelineV2
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

class DumpLearningData(BasePipelineV2):
    """Records to feature proto pipeline."""

    def __init__(self):
        self.src_prefixs = [
            'modules/planning/cleaned_data',
        ]
        self.dest_prefix = 'modules/planning/learning_data'

    def run_test(self):
        """Run"""
        self.src_prefixs = [
            '/apollo/data/cleaned_data',
        ]
        self.dest_prefix = '/apollo/data/learning_data'

        src_dirs_set = set([])
        for prefix in self.src_prefixs:
            for root, dirs, files in os.walk(prefix):
                for file in files:
                    src_dirs_set.add(root)

        processed_records = self.to_rdd(src_dirs_set).map(self.process_record)

        logging.info('Processed {}/{} records'.format(processed_records.count(),
                                                      len(src_dirs_set)))
        return 0

    def run(self):
        """Run"""
        # for prefix in self.src_prefixs:
        #    logging.info(self.our_storage().list_files(prefix))

        records_rdd = BasePipelineV2.SPARK_CONTEXT.union([
            self.to_rdd(self.our_storage().list_files(prefix))
                .filter(record_utils.is_record_file)
                .map(os.path.dirname)
                .distinct()
            for prefix in self.src_prefixs])

        processed_records = records_rdd.map(self.process_record)

        logging.info('Processed {} records'.format(processed_records.count()))

    def process_record(self, src_record_fn):
        """ Process Records """
        src_record_fn_elements = src_record_fn.split("/")
        timestamp = src_record_fn_elements[-2]
        dest_record_dir_elements = []
        if (not timestamp.startswith("ver_")):
            # casual format folder layout
            timestamp = "ver_" + datetime.date.today().strftime("%Y%m%d_%H%M%S") + "/"
            dest_record_dir_elements.append(self.dest_prefix)
            dest_record_dir_elements.append(timestamp)
        else:
            # pipeline format folder layout
            for dir_name in src_record_fn_elements:
                if dir_name == "cleaned_data":
                    dest_record_dir_elements.append("learning_data");
                else:
                    dest_record_dir_elements.append(dir_name);
        dest_record_dir = "/".join(dest_record_dir_elements)

        file_utils.makedirs(os.path.dirname(dest_record_dir))

        """Call planning C+pro+ code."""
        map_name = "sunnyvale"
        command = (
            'cd /apollo && sudo bash '
            'modules/tools/planning/data_pipelines/scripts/'
            'records_to_data_for_learning.sh '
            '"{}" "{}" "{}"'.format(src_record_fn, dest_record_dir, map_name))
        if os.system(command) == 0:
            logging.info('Successfully processed {} to {}'.format(src_record_fn,
                                                                  dest_record_dir))
            return 1
        else:
            logging.error('Failed to process {} to {}'.format(src_record_fn,
                                                              dest_record_dir))
        return 0

if __name__ == '__main__':
    DumpLearningData().main()
