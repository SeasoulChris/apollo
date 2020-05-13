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
from fueling.control.dynamic_model_2_0.conf.model_conf import \
     segment_index, feature_config, input_index, output_index
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.control.dynamic_model_2_0.feature_extraction.feature_extraction_utils as \
       feature_utils_2_0 
import fueling.control.features.feature_extraction_utils as feature_utils
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

flags.DEFINE_integer('percentile', 50, 'percentile of the data.')


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
            self.to_rdd(self.our_storage().list_files(src_prefix))
            # RDD(hdf5 files)
            .filter(spark_op.filter_path(['*.hdf5'])))

        categories = spark_helper.cache_and_log('CalculatedCategories',
            # RDD(hdf5 files)
            hdf5_files
            # PairRDD(hdf5 file, hdf5 file)
            .keyBy(lambda hdf5_file: hdf5_file)
            # PairRDD(hdf5 file, segment)
            .mapValues(label_generation.generate_segment)
            # PairRDD(category_id, (hdf5 file, segment))
            .map(lambda file_segment: (self.calculate_category_id(file_segment[1]),
                (file_segment[0], file_segment[1])))
            # PairRDD(category_id, (hdf5 file, segment)s)
            .groupByKey())

        partitions = int(os.environ.get('APOLLO_EXECUTORS', 15))
        (categories
            .repartition(partitions)
            .foreach(lambda cate: self.process_file(cate, src_prefix, dst_prefix, model_path)))


    def process_file(self, category, src_prefix, dst_prefix, model_path):
        """Process category group and generate h5 files"""
        category_id, segments = category
        print(F'category id: {category_id}, segment length: {len(segments)}')

        dst_prefix = os.path.join(dst_prefix, category_id)
        segmnets_count_for_each_category = feature_config['SAMPLE_SIZE']
        for hdf5_file, segment in segments:
            # Get destination file
            dst_dir = os.path.dirname(hdf5_file).replace(src_prefix, dst_prefix, 1)
            file_utils.makedirs(dst_dir)
            dst_h5_file = os.path.join(dst_dir, os.path.basename(hdf5_file).replace('hdf5', 'h5'))

            self.write_single_h5_file(dst_h5_file, segment, model_path)

            segmnets_count_for_each_category -= 1
            if segmnets_count_for_each_category == 0:
                break


    def write_single_h5_file(self, dst_h5_file, segment, model_path):
        """Write segment into a single h5 file"""
        model_path = file_utils.fuel_path(model_path)
        input_segment, output_segment = label_generation.generate_gp_data(model_path, segment)

        with h5py.File(dst_h5_file, 'w') as h5_file:
            h5_file.create_dataset('input_segment', data=input_segment)
            h5_file.create_dataset('output_segment', data=output_segment)
        logging.info(F'successfully write into destination h5 file: {dst_h5_file}')


    def calculate_category_id(self, segment):
        """Calculate category id by given segment"""
        # Get a list of values corresponding to 'speed', 'throttle', 'steering', 'brake'
        features = feature_utils_2_0.filter_dimensions(segment, self.FLAGS.get('percentile'))
        funcs = [feature_utils.gen_speed_key,
                 feature_utils.gen_throttle_key,
                 feature_utils.gen_steering_key,
                 feature_utils.gen_brake_key]
        category_id = ''
        for idx, (feature_name, feature_value) in enumerate(features):
            if feature_name != 'speed':
                feature_value *= 100.0
            category_id += str(funcs[idx](feature_value))
        return category_id 


if __name__ == '__main__':
    PipelineLabelGenerator().main()
