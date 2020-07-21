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

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model_2_0.conf.model_conf import \
    segment_index, feature_config, label_config, input_index, output_index
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.redis_utils as redis_utils
import fueling.common.spark_helper as spark_helper
import fueling.common.spark_op as spark_op
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
        REDIS_KEY = 'control.dm20.label.counters'
        timestr = time.strftime('%Y-%m-%d-%H')
        src_prefix = flags.FLAGS.DM20_features
        dst_prefix = os.path.join(flags.FLAGS.DM20_labeled_features, timestr)
        if self.is_local():
            logging.info('at local')
            src_prefix = 'local_test/features'
            dst_prefix = os.path.join('local_test/labeled_data', timestr)
        model_path = flags.FLAGS.DM10_forward_mlp_model

        logging.info(F'src: {src_prefix}, dst: {dst_prefix}, model path: {model_path}')
        redis_utils.redis_remove_key(REDIS_KEY)
        redis_utils.redis_extend_dict(REDIS_KEY, {})
        hdf5_files = spark_helper.cache_and_log(
            'OutputRecords',
            # RDD(end_files)
            self.to_rdd(self.our_storage().list_files(src_prefix))
            # RDD(hdf5 files)
            .filter(spark_op.filter_path(['*.hdf5'])))

        all_segments = (
            hdf5_files
            # PairRDD(hdf5 file, hdf5 file)
            .keyBy(lambda hdf5_file: hdf5_file)
            # PairRDD(hdf5 file, segment)
            .mapValues(label_generation.generate_segment)
            # PairRDD(hdf5 file, sub_segment)
            .flatMapValues(self.get_sub_segments)
            # PairRDD(hdf5 file, sub_segment)
            .map(lambda file_segment:
                 self.process_file(file_segment, src_prefix, dst_prefix,
                                   model_path, REDIS_KEY))
            .count())

        logging.info(F'Done with control label generation, {all_segments} segments generated')

    def process_file(self, file_segment, src_prefix, dst_prefix, model_path, redis_key):
        """Process category group and generate h5 files"""
        hdf5_file, segment = file_segment
        category_id = self.calculate_category_id(segment)
        dst_prefix = os.path.join(dst_prefix, category_id)
        dst_dir = os.path.dirname(hdf5_file).replace(src_prefix, dst_prefix, 1)
        file_utils.makedirs(dst_dir)
        category_seq = self.get_category_seq(redis_key, category_id)
        if category_seq is None or not str.isdigit(category_seq):
            logging.fatal(F'errors occurred when get {category_seq} from Redis for {category_id}')
        if int(category_seq) >= label_config['SAMPLE_SIZE']:
            logging.info(F'got enough segments for category {category_id}')
            return 0
        dst_h5_file = os.path.join(dst_dir, F'{os.path.basename(hdf5_file)[:-5]}_{category_seq}.h5')
        self.write_single_h5_file(dst_h5_file, segment, model_path)
        return 1

    def get_category_seq(self, redis_key, category_id):
        """Get how many items for certain category from Redis"""
        redis_instance = redis_utils.get_redis_instance()
        category_seq = None
        lock_key = F'{category_id}-lock'
        with redis_instance.lock(lock_key):
            if not redis_instance.hexists(redis_key, category_id):
                logging.info(F'{category_id} does not exist, create new one')
                redis_instance.hmset(redis_key, {category_id: 0})
            redis_instance.hincrby(redis_key, category_id)
            category_seq = redis_instance.hget(redis_key, category_id)
        logging.log_every_n(logging.INFO, F'category seq {category_seq} for {category_id}', 10)
        return category_seq

    def get_sub_segments(self, segment):
        """Split a two dimensional numpy array (N x 23) into multiple pieces"""
        sub_segments = [segment[idx: idx + label_config['LABEL_SEGMENT_LEN']]
                        for idx in range(
                            0,
                            segment.shape[0] - label_config['LABEL_SEGMENT_LEN'] + 1,
                            label_config['LABEL_SEGMENT_STEP'])]
        logging.info(F'got {len(sub_segments)} sub segments from {segment.shape[0]} frames')
        return sub_segments

    def write_single_h5_file(self, dst_h5_file, segment, model_path):
        """Write segment into a single h5 file"""
        model_path = file_utils.fuel_path(model_path)
        input_segment, output_segment = label_generation.generate_gp_data(model_path, segment)
        with h5py.File(dst_h5_file, 'w') as h5_file:
            h5_file.create_dataset('input_segment', data=input_segment)
            h5_file.create_dataset('output_segment', data=output_segment)
        logging.log_every_n(logging.INFO, F'wrote single file {dst_h5_file}', 10)

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
