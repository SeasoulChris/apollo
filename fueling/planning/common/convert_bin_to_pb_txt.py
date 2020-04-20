#!/usr/bin/env python
import os
import time

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
from modules.planning.proto import learning_data_pb2


class ConvertBinToPBTxt(BasePipeline):
    """Records to feature proto pipeline."""

    def __init__(self):
        self.src_file = 'modules/planning/learning_data/2019-10-17-13-36-41/learning_data.0.bin'
        # self.src_file = '/apollo/data/learning_data/task_2/learning_data.0.bin'
        # self.src_file = '/apollo/data/output_data/learning_data.0.bin.future_status.npy'

    def run_test(self):
        pass

    def run(self):
        """Run"""
        self.process_file(self.src_file)
        logging.info('Processed: {}'.format(self.src_file))

    def process_file(self, src_file):
        """ Process File"""
        dest_dir, dest_filename = os.path.split(src_file)
        dest_filename += ".txt.tmp"
        dest_file = src_file + ".txt.tmp"

        logging.info("src [{}]; dest[{}]".format(src_file, dest_file))

        pb = learning_data_pb2.LearningData()
        pb = proto_utils.get_pb_from_bin_file(src_file, pb)
        proto_utils.write_pb_to_text_file(pb, dest_file)

        return 0


if __name__ == '__main__':
    ConvertBinToPBTxt().main()
