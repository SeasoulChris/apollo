#!/usr/bin/env python
from datetime import date
import glob
import os

import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.control.dynamic_model.conf.model_config import feature_extraction, feature_config
import fueling.common.h5_utils as h5_utils
import fueling.common.logging as logging
import fueling.control.dynamic_model.conf.model_config as model_config
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

# parameters
INTER_FOLDER = feature_extraction['inter_result_folder']
OUTPUT_FOLDER = feature_extraction['uniform_output_folder']
SAMPLE_SIZE = feature_extraction['sample_size']


def pick_sample(list_of_segment, sample_size):
    counter = 0
    sample_list = []
    for segment in list_of_segment:
        add_size = segment.shape[0]  # row, data points
        if counter + add_size <= sample_size:
            sample_list.append(segment)
            counter += add_size
        elif counter <= sample_size:
            to_add_size = max(sample_size - counter, model_config.feature_config['sequence_length'])
            sample_list.append(segment[0:to_add_size, :])
            return (sample_list, (counter + to_add_size))
    return (sample_list, counter)


def write_to_file(target_prefix, elem):
    (vehicle, key), list_of_segment = elem
    total_number = len(list_of_segment)
    file_dir = os.path.join(target_prefix, vehicle, key)
    counter = 1
    for data in list_of_segment:
        file_name = str(counter).zfill(4) + "_of_" + str(total_number).zfill(4)
        h5_utils.write_h5_single_segment(data, file_dir, file_name)
        counter += 1
    return total_number, file_dir, file_name


class UniformSet(BasePipeline):

    def run_test(self):
        """Run test."""
        # folder
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/generated/'
        target_prefix = '/apollo/modules/data/fuel/testdata/control/generated_uniform'

        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        IS_BACKWARD = self.FLAGS.get('is_backward')

        if IS_BACKWARD:
            origin_prefix = os.path.join(origin_prefix, job_owner, 'backward', job_id)
            target_prefix = os.path.join(target_prefix, job_owner, 'backward', job_id)
        else:
            origin_prefix = os.path.join(origin_prefix, job_owner, 'forward', job_id)
            target_prefix = os.path.join(target_prefix, job_owner, 'forward', job_id)

        # get vehicles
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_prefix])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(origin_prefix, vehicle_type)))
        logging.info(origin_vehicle_dir.collect())

        # hdf5 files
        feature_dir = spark_helper.cache_and_log(
            'hdf5_files',
            origin_vehicle_dir
            # PairRDD(vehicle, hdf5 file)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*.hdf5'))))

        self.run(feature_dir, origin_vehicle_dir, target_prefix)

    def run_prod(self):
        """Run prod."""

        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        IS_BACKWARD = self.FLAGS.get('is_backward')
        bos_client = self.our_storage()

        # intermediate result folder
        if IS_BACKWARD:
            origin_prefix = os.path.join(INTER_FOLDER, job_owner, 'backward', job_id)
            target_prefix = os.path.join(OUTPUT_FOLDER, job_owner, 'backward', job_id)
        else:
            origin_prefix = os.path.join(INTER_FOLDER, job_owner, 'forward', job_id)
            target_prefix = os.path.join(OUTPUT_FOLDER, job_owner, 'forward', job_id)

        origin_dir = bos_client.abs_path(origin_prefix)
        logging.info('origin dir: %s' % origin_dir)

        # output folder
        target_dir = bos_client.abs_path(target_prefix)
        logging.info('target dir: %s' % target_dir)

        # use prefix to list files
        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_dir])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        # PairRDD(vehicle, hdf5 files)
        hdf5_files = spark_helper.cache_and_log(
            'hdf5_files',
            origin_vehicle_dir
            .flatMapValues(lambda path: self.our_storage().list_files(path, '.hdf5')))

        self.run(hdf5_files, origin_vehicle_dir, target_dir)

    def run(self, feature_dir, origin_vehicle_conf_dir, target_dir):
        def _generate_key(elements):
            vehicle, hdf5 = elements
            file_name = os.path.basename(hdf5)
            key = file_name.split('_', 1)[0]
            return (vehicle, key), hdf5

        categorized_segments = spark_helper.cache_and_log(
            'categorized_segments',
            # PairRDD(vehicle, .hdf5 files with absolute path)
            feature_dir
            # PairRDD((vehicle, key), file_path)
            .map(_generate_key)
            # PairRDD((vehicle, key), segments)
            .mapValues(h5_utils.read_h5)
            # PairRDD((vehicle, key), segments RDD)
            .groupByKey()
            # PairRDD((vehicle, key), list of segments)
            .mapValues(list))

        spark_helper.cache_and_log(
            'sampled_segments',
            # PairRDD((vehicle, key), list of segments)
            categorized_segments
            # PairRDD((vehicle, key), (sampled segments, counter))
            .mapValues(lambda samples: pick_sample(samples, SAMPLE_SIZE))
            # PairRDD(((vehicle, key), (sampled segments, counter))
            .mapValues(lambda segment_counter: segment_counter[0])
            # RDD(segment_length)
            .map(lambda elem: write_to_file(target_dir, elem)))


if __name__ == '__main__':
    UniformSet().main()
