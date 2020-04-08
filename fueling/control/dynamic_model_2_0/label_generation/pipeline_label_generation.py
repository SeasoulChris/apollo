#!/usr/bin/env python
import operator
import os
import time

# disable GPU for local test
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import load_model
import h5py
import numpy as np

from absl import flags
import pyspark_utils.op as spark_op

from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config, input_index, output_index
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.file_utils as file_utils
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

    def run(self):
        timestr = time.strftime('%Y%m%d')
        if self.is_local():
            self.src_dir_prefix = 'local_test/features'
            self.dst_dir_prefix = os.path.join('local_test/labeled_data', timestr)
        else:
            self.src_dir_prefix = flags.FLAGS.DM20_features
            self.dst_dir_prefix = os.path.join(flags.FLAGS.DM20_labeled_features, timestr)
        self.model_path = flags.FLAGS.DM10_forward_mlp_model
        logging.info(self.src_dir_prefix)
        logging.info(self.dst_dir_prefix)
        logging.info(self.model_path)
        hdf5_files_rdd = (
            self.to_rdd(self.our_storage().list_files(self.src_dir_prefix))
            .filter(spark_op.filter_path(['*.hdf5']))
        )
        # logging.info(hdf5_files_rdd.collect())
        self.run_internal(hdf5_files_rdd)

    def run_internal(self, hdf5_files_rdd):
        hdf5_files_rdd = hdf5_files_rdd.keyBy(lambda src_file: os.path.dirname(src_file))
        # logging.info(hdf5_files_rdd.collect())
        result = hdf5_files_rdd.mapValues(lambda hdf5_src_file: self.process_file(
            hdf5_src_file, self.src_dir_prefix, self.dst_dir_prefix, self.model_path)).cache()
        # logging.info(results.collect())
        if result.isEmpty():
            logging.info("Nothing to be processed.")
            return
        logging.info('Processed {}/{} tasks'.format(result.values().reduce(operator.add), result.count()))

    @staticmethod
    def process_file(hdf5_src_file, src_dir_prefix, dst_dir_prefix, model_path):
        model_path = file_utils.fuel_path(model_path)
        # target file name
        logging.info(f'processing file: {hdf5_src_file}')
        src_file_name = os.path.basename(hdf5_src_file)
        src_file_id = src_file_name.split('.')[0]
        # logging.info(f'src_file_id: {src_file_id}')
        src_dir = os.path.dirname(hdf5_src_file)
        dst_dir = src_dir.replace(src_dir_prefix, dst_dir_prefix)
        # logging.info(f'dst_dir: {dst_dir}')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        file_name = os.path.join(dst_dir, src_file_id + '.h5')
        # logging.info(f'file_name: {file_name}')
        # commentted for now
        # try:
        segment = label_generation.generate_segment(hdf5_src_file)
        input_segment, output_segment = label_generation.generate_gp_data(model_path, segment)
        with h5py.File(file_name, 'w') as h5_file:
            h5_file.create_dataset('input_segment', data=input_segment)
            h5_file.create_dataset('output_segment', data=output_segment)
            logging.info('Successfully labeled {}'.format(hdf5_src_file))
        return 1
        # except BaseException:
        #     logging.error('Failed to process {}'.format(hdf5_src_file))
        #     return 0


if __name__ == '__main__':
    PipelineLabelGenerator().main()
