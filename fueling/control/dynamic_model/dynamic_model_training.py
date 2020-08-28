#!/usr/bin/env python

import collections
import os

import glob
import numpy as np

from fueling.common.base_pipeline import BasePipeline
from fueling.common.job_utils import JobUtils
from fueling.control.dynamic_model.conf.model_config import task_config
import fueling.common.email_utils as email_utils
import fueling.common.logging as logging
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction
import fueling.control.dynamic_model.data_generator.training_data_generator as data_generator
import fueling.control.dynamic_model.model_factory.lstm_keras as lstm_keras
import fueling.control.dynamic_model.model_factory.mlp_keras as mlp_keras


class DynamicModelTraining(BasePipeline):

    def run_test(self):
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        is_backward = self.FLAGS.get('is_backward')
        is_holistic = self.FLAGS.get('is_holistic')
        data_dir = '/fuel/testdata/control/generated_uniform'
        if is_backward:
            training_data_path = os.path.join(data_dir, job_owner, 'backward', job_id)
        else:
            training_data_path = os.path.join(data_dir, job_owner, 'forward', job_id)
        model_dir = '/fuel/testdata/control/learning_based_model'
        output_dir = os.path.join(model_dir, 'dynamic_model_output')

        vehicles = multi_vehicle_utils.get_vehicle(training_data_path)
        logging.info(f'vehicles = {vehicles}')
        # run test as a vehicle ID
        for vehicle in vehicles:
            self.execute_task(vehicle, training_data_path, output_dir, is_backward, is_holistic)

    def run(self):
        # initialize input/output dirs
        job_owner = self.FLAGS.get('job_owner')
        job_id = self.FLAGS.get('job_id')
        is_backward = self.FLAGS.get('is_backward')
        is_holistic = self.FLAGS.get('is_holistic')

        # Check partner storage for online service
        object_storage = self.partner_storage() or self.our_storage()

        data_dir = task_config['uniform_output_folder']
        if is_backward:
            data_prefix = os.path.join(data_dir, job_owner, 'backward', job_id)
        else:
            data_prefix = os.path.join(data_dir, job_owner, 'forward', job_id)
        training_data_path = object_storage.abs_path(data_prefix)

        output_dir = object_storage.abs_path(task_config['model_output_folder'])

        # get vehicles
        vehicles = multi_vehicle_utils.get_vehicle(training_data_path)
        vehicles_num = len(vehicles)
        logging.info(f'vehicles: {vehicles}')

        # prepare for email notification
        SummaryTuple = collections.namedtuple(
            'Summary',
            ['Vehicle', 'Input_Data_Path', 'Output_Model_Path', 'Is_Backward', 'Is_Holistic'])
        messages = []

        # run proc as a vehicle ID
        vehicle_count = 0
        for vehicle in vehicles:
            vehicle_count += 1
            self.execute_task(vehicle, training_data_path, output_dir, is_backward, is_holistic)
            messages.append(SummaryTuple(
                Vehicle=vehicle, Input_Data_Path=training_data_path, Output_Model_Path=output_dir,
                Is_Backward=is_backward, Is_Holistic=is_holistic))
            JobUtils(job_id).save_job_progress(45 + (95 - 45) * vehicle_count / vehicles_num)

        # send email notification
        title = F'Control Dynamic Model Training Results For {len(vehicles)} Vehicles'
        # TODO(longtao): add all receivers after testing email notification
        email_utils.send_email_info(title, messages, email_utils.DATA_TEAM)
        JobUtils(job_id).save_job_progress(100)

    def execute_task(self, vehicle, training_data_path, output_dir,
                     is_backward=False, is_holistic=False):
        # vehicle dir
        vehicle_dir = os.path.join(training_data_path, vehicle)
        # model output dir
        model_output_dir = os.path.join(output_dir, vehicle)
        logging.info(f'vehicle_dir: {vehicle_dir}')
        logging.info(f'model_output_dir: {model_output_dir}')
        # RDD hd5_dataset
        hd5_files_path = glob.glob(os.path.join(vehicle_dir, '*/*.hdf5'))
        # logging.info('hd5_files_path = {}'.format(hd5_files_path))
        # for file in files_path:
        training_dataset_rdd = self.to_rdd(hd5_files_path)
        self.run_internal(training_dataset_rdd, model_output_dir, is_backward, is_holistic)

    def run_internal(self, training_dataset_rdd, output_dir, is_backward=False, is_holistic=False):
        data = (
            # RDD(absolute_file_path)
            training_dataset_rdd
            # RDD(training_data_segment)
            .map(feature_extraction.generate_segment)
            # RDD(training_data_segment), which is valid.
            .filter(lambda segment: segment is not None)
            # RDD(training_data_segment), smoothing input features.
            .map(feature_extraction.feature_preprocessing)
            # RDD(training_data_segment), which is valid after feature_preprocessing.
            .filter(lambda segment: segment is not None)
            # RDD('mlp_data|lstm_data', (input, output)).
            .flatMap(lambda segment: data_generator.generate_training_data(segment, is_holistic))
            # RDD('mlp_data|lstm_data', (input, output)), which is valid.
            .filter(lambda data: data is not None)
            # RDD('mlp_data|lstm_data', (input, output)), with unique keys.
            .reduceByKey(lambda data_1, data_2: (np.concatenate((data_1[0], data_2[0]), axis=0),
                                                 np.concatenate((data_1[1], data_2[1]), axis=0)))
            .cache())

        param_norm = (
            # RDD('mlp_data', (input, output))
            data.filter(lambda key_value: key_value[0] == 'mlp_data')
            # RDD('mlp_data', param_norm)
            .mapValues(lambda input_output: feature_extraction.get_param_norm(input_output[0],
                                                                              input_output[1]))
            # param_norm
            .values()
            .first())
        logging.info(f'Param Norm = {param_norm}')

        def _train(data_item, is_backward=False, is_holistic=False):
            key, (input_data, output_data) = data_item
            if key == 'mlp_data':
                mlp_keras.mlp_keras(
                    input_data,
                    output_data,
                    param_norm,
                    output_dir,
                    is_backward,
                    is_holistic)
            elif key == 'lstm_data':
                lstm_keras.lstm_keras(
                    input_data,
                    output_data,
                    param_norm,
                    output_dir,
                    'lstm_two_layer',
                    is_backward,
                    is_holistic)

        data.foreach(lambda data_item: _train(data_item, is_backward, is_holistic))


if __name__ == '__main__':
    DynamicModelTraining().main()
