#!/usr/bin/env python

import operator
import os
import time

# disable GPU for local test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from absl import flags
from keras.models import load_model
import h5py
import numpy as np
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config, input_index, output_index
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.control.dynamic_model_2_0.label_generation.label_generation as label_generation


flags.DEFINE_string('DM20_features',
                    'modules/control/dynamic_model_2_0/features',
                    'input data directory')
flags.DEFINE_string('DM20_labeled_features',
                    'modules/control/dynamic_model_2_0/labeled_data',
                    'output data directory')
flags.DEFINE_string('DM10_forward_mlp_model',
                    'fueling/control/dynamic_model_2_0/label_generation/mlp_model',
                    'mlp forward model directory')


class PipelineLabelGenerator(BasePipeline):
    """Labeling pipeline"""

    def run(self):
        """Run."""
        timestr = time.strftime('%Y-%m-%d-%H')
        src_prefix = flags.FLAGS.DM20_features
        dst_prefix = os.path.join(flags.FLAGS.DM20_labeled_features, timestr)
        if self.is_local():
            logging.info('at local')
            src_prefix = 'local_test/features'
            dst_prefix = os.path.join('local_test/labeled_data', timestr)
        model_path = flags.FLAGS.DM10_forward_mlp_model

        logging.info(F'src: {src_prefix}, dst: {dst_prefix}, model path: {model_path}')

        hdf5_files = spark_helper.cache_and_log('OutputRecords',
                                                # RDD(end_files)
                                                self.to_rdd(
                                                    self.our_storage().list_files(src_prefix))
                                                # RDD(hdf5 files)
                                                .filter(spark_op.filter_path(['*.hdf5'])))

        completed_file_count = (hdf5_files
                                # RDD(hdf5 files)
                                .map(lambda hdf5_file: self.process_file(hdf5_file, src_prefix, dst_prefix, model_path))
                                .reduce(operator.add))

        logging.info(F'processed {completed_file_count}/{hdf5_files.count()} hdf5 files')

    @staticmethod
    def process_file(hdf5_file, src_prefix, dst_prefix, model_path):
        """Process one single hdf5 file and generate h5 file"""
        logging.info(f'processing file: {hdf5_file}')
        model_path = file_utils.fuel_path(model_path)

        dst_dir = os.path.dirname(hdf5_file).replace(src_prefix, dst_prefix, 1)
        file_utils.makedirs(dst_dir)

        segment = label_generation.generate_segment(hdf5_file)
        input_segment, output_segment = label_generation.generate_gp_data(model_path, segment)

        # replace previous .hdf5 extension as h5
        dst_h5_file = os.path.join(
            dst_dir, os.path.basename(hdf5_file).replace('hdf5', 'h5'))
        logging.info(dst_h5_file)
        with h5py.File(dst_h5_file, 'w') as h5_file:
            h5_file.create_dataset('input_segment', data=input_segment)
            h5_file.create_dataset('output_segment', data=output_segment)
            logging.info('Successfully labeled {}'.format(hdf5_file))
        return 1


if __name__ == '__main__':
    PipelineLabelGenerator().main()
