#!/usr/bin/env python
import operator
import os
# disable GPU for local test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import load_model
import h5py
import numpy as np

import pyspark_utils.op as spark_op

from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config, input_index, output_index
from fueling.common.base_pipeline import BasePipeline
import fueling.control.dynamic_model_2_0.label_generation.label_generation as label_generation
import fueling.common.logging as logging


# bos path
SRC_DIR_PREFIX = 'modules/control/dynamic_model_2_0/features'
DST_DIR_PREFIX = 'modules/control/dynamic_model_2_0/labeled_data'
MODEL_PATH = 'modules/control/dynamic_model_2_0/mlp_model/forward'

# # local path
# MODEL_PATH = 'fueling/control/dynamic_model_2_0/label_generation/mlp_model'
# SRC_DIR_PREFIX = 'local_test/features'
# DST_DIR_PREFIX = 'local_test/labeled_data'


class PipelineLabelGenerator(BasePipeline):
    def run(self):
        hdf5_files_rdd = (
            self.to_rdd(self.our_storage().list_files(SRC_DIR_PREFIX))
            .filter(spark_op.filter_path(['*.hdf5']))
        )
        logging.info(hdf5_files_rdd.collect())
        self.run_internal(hdf5_files_rdd)

    def run_internal(self, hdf5_files_rdd):
        hdf5_files_rdd = hdf5_files_rdd.keyBy(lambda src_file: os.path.dirname(src_file))
        logging.info(hdf5_files_rdd.collect())
        result = hdf5_files_rdd.mapValues(self.process_file).cache()
        if result.isEmpty():
            logging.info("Nothing to be processed.")
            return
        logging.info('Processed {}/{} tasks'.format(result.values().reduce(operator.add), result.count()))

    @staticmethod
    def process_file(hdf5_src_file):
        # target file name
        logging.info(f'processing file: {hdf5_src_file}')
        src_file_name = os.path.basename(hdf5_src_file)
        src_file_id = src_file_name.split('.')[0]
        logging.info(f'src_file_id: {src_file_id}')
        src_dir = os.path.dirname(hdf5_src_file)
        dst_dir = src_dir.replace(SRC_DIR_PREFIX, DST_DIR_PREFIX)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        file_name = os.path.join(dst_dir, src_file_id + '.h5')
        logging.info(f'file_name: {file_name}')
        try:
            segment = label_generation.generate_segment(hdf5_src_file)
            input_segment, output_segment = label_generation.generate_gp_data(MODEL_PATH, segment)
            with h5py.File(file_name, 'w') as h5_file:
                h5_file.create_dataset('input_segment', data=input_segment)
                h5_file.create_dataset('output_segment', data=output_segment)
                logging.info('Successfully labeled {}'.format(hdf5_src_file))
            return 1
        except BaseException:
            logging.error('Failed to process {}'.format(hdf5_src_file))
            return 0


if __name__ == '__main__':
    PipelineLabelGenerator().main()
