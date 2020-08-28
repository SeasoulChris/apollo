#!/usr/bin/env python

import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction
from fueling.control.dynamic_model.conf.model_config import task_config
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.common.multi_vehicle_plot_utils as multi_vehicle_plot_utils
import fueling.common.spark_helper as spark_helper
import fueling.common.logging as logging
from fueling.common.job_utils import JobUtils
from fueling.common.base_pipeline import BasePipeline
import os
import time

import glob
import matplotlib
matplotlib.use('Agg')


class DynamicModelDatasetDistribution(BasePipeline):

    def run_test(self):
        # hdf5 data directory
        data_dir = '/fuel/testdata/control/learning_based_model'
        file_dir = os.path.join(
            data_dir, 'hdf5_training/Mkz7/UniformDistributed/forward/*/*/*.hdf5')
        # data_dir = '/fuel/testdata/control/generated/Mkz7/SampleSet'
        # training_dataset = [os.path.join(data_dir, 'hdf5_training/training_test.hdf5')]

        # file path to save visualization results
        output_dir = os.path.join(data_dir, 'evaluation_result')
        timestr = time.strftime('%Y%m%d-%H%M%S')
        file_name = ('dataset_distribution_%s.pdf' % timestr)
        result_file = os.path.join(output_dir, file_name)

        hdf5_file_list = self.to_rdd(glob.glob(file_dir))
        print(hdf5_file_list.collect())
        self.run_internal('Mkz7', hdf5_file_list, result_file)

    def run(self):
        # initialize input/output dirs
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        is_backward = self.FLAGS.get('is_backward')

        our_storage = self.our_storage()

        data_dir = task_config['sample_output_folder']
        if is_backward:
            data_prefix = os.path.join(data_dir, job_owner, 'backward', job_id)
        else:
            data_prefix = os.path.join(data_dir, job_owner, 'forward', job_id)
        visual_data_prefix = our_storage.abs_path(data_prefix)

        # get vehicles
        vehicles = multi_vehicle_utils.get_vehicle(visual_data_prefix)
        logging.info(f'Vehicles: {vehicles}')

        # run proc as a vehicle ID
        for vehicle in vehicles:
            # list hdf5 files from sample set
            visual_data_path = os.path.join(visual_data_prefix, vehicle)
            hdf5_file_list = self.to_rdd(self.our_storage().list_files(visual_data_path, '.hdf5'))
            logging.info(f'Visual data file: {hdf5_file_list.collect()}')
            # define visual result file name
            output_dir = os.path.join(visual_data_path, 'visual_result')
            os.mkdir(output_dir)
            result_file = os.path.join(output_dir, 'dynamic_model_sample_dataset_distribution.pdf')
            logging.info(f'Visual result file: {result_file}')
            # generate visual result file
            self.run_internal(vehicle, hdf5_file_list, result_file)
        JobUtils(job_id).save_job_progress(45)

    def run_internal(self, vehicle, hdf5_file_list, result_file):

        data = spark_helper.cache_and_log(
            'plots',
            hdf5_file_list.keyBy(lambda _: vehicle)
            .groupByKey()
            .mapValues(feature_extraction.generate_segment_from_list))

        spark_helper.cache_and_log(
            'plots',
            data
            .mapValues(lambda features: multi_vehicle_plot_utils
                       .plot_dynamic_model_feature_hist(features, result_file)))


if __name__ == '__main__':
    DynamicModelDatasetDistribution().main()
