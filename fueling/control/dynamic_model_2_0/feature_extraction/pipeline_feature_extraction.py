#!/usr/bin/python

import glob
import os


from absl import flags
import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op


from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model_2_0.feature_extraction.feature_extractor import DynamicModel20FeatureExtractor
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils


flags.DEFINE_string('input_folder',
                    'modules/control/data/records/Mkz7/2019-04-30',
                    'golden set record directory')

flags.DEFINE_string('output_folder',
                    'modules/control/dynamic_model_2_0/golden_set/features/2019-04-30',
                    'golden set feature directory')


# flags.DEFINE_string('input_folder',
#                     'fueling/control/dynamic_model_2_0/testdata/golden_set/2_3',
#                     'golden set input directory')

# flags.DEFINE_string('output_folder',
#                     'fueling/control/dynamic_model_2_0/testdata/golden_set_output',
#                     'golden set output directory')


class PipelineFeatureExtraction(BasePipeline):
    def __init__(self):
        super().__init__()
        self.data_length_pre_frame = 100
        # self.input_folder = flags.FLAGS.input_folder

    def run(self):
        input_folder = flags.FLAGS.input_folder
        output_folder = flags.FLAGS.output_folder
        # get subfolder of all records
        task = (
            self.to_rdd(self.our_storage().list_files(input_folder))
            .filter(spark_op.filter_path(['*.recover']))).cache()
        logging.info(task.first())

        # make dirs
        dst_dir_rdd = (task
                       # RDD(dir)
                       .map(os.path.dirname)
                       # RDD(unique dir)
                       .distinct()
                       # RDD(dst dir)
                       .map(lambda src_dir: src_dir.replace(input_folder, output_folder, 1)))
        logging.info(dst_dir_rdd.collect())
        dst_dir_rdd.foreach(file_utils.makedirs)

        # process files
        task.foreach(lambda file_name: self.process_file(
            file_name, input_folder, output_folder))
        logging.info(task.count())

    @staticmethod
    def process_file(file_name, input_prefix, output_prefix):
        feature_extractor = DynamicModel20FeatureExtractor(file_name, secondary_file=None,
                                                           origin_prefix=input_prefix, target_prefix=output_prefix)
        feature_extractor.extract_data_from_record()
        feature_extractor.data_pairing()
        feature_extractor.extract_features()


if __name__ == '__main__':
    PipelineFeatureExtraction().main()
