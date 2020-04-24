#!/usr/bin/env python
import os
import time

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from modules.planning.proto import learning_data_pb2


class ConvertBinToPBTxt(BasePipeline):
    def __init__(self):
        self.src_dir_prefixs = [
            'modules/planning/output_data/test/',
        ]

    def run(self):
        """Run Test"""
        self.src_dir_prefixs = [
            '/fuel/data/output_data/test/',
        ]

        src_dirs_set = set([])
        for prefix in self.src_dir_prefixs:
            for root, dirs, files in os.walk(prefix):
                for file in files:
                    src_dirs_set.add(root)

        processed_dirs = self.to_rdd(src_dirs_set).map(self.process_dir)

        logging.info('Processed {}/{} folders'.format(processed_dirs.count(),
                                                      len(src_dirs_set)))
        return 0
    '''
    def run(self):
        """Run"""
        records_rdd = BasePipeline.SPARK_CONTEXT.union([
            self.to_rdd(self.our_storage().list_files(prefix))
                .filter(record_utils.is_record_file)
                .map(os.path.dirname)
                .distinct()
            for prefix in self.src_dir_prefixs])

        processed_dirs = records_rdd.map(self.process_dir)

        logging.info('Processed {} folders'.format(processed_dir.count()))
    '''
    def process_dir(self, src_dir):
        for filename in os.listdir(src_dir):
            if filename.endswith(".bin"):
                src_file = os.path.join(src_dir, filename)
                dest_file = src_file + ".txt"
                pb = learning_data_pb2.LearningData()
                pb = proto_utils.get_pb_from_bin_file(src_file, pb)
                proto_utils.write_pb_to_text_file(pb, dest_file)

                logging.info("src [{}]; dest[{}]".format(src_file, dest_file))
        return 0


if __name__ == '__main__':
    ConvertBinToPBTxt().main()
