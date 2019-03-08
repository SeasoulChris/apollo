#!/usr/bin/env python
import glob

import h5py
import numpy as np
import pyspark_utils.helper as spark_helper

import fueling.control.offline_evaluator.trajectory_visualization as trajectory_visualization
import fueling.control.training_models.mlp_keras as mlp_keras

from fueling.common.base_pipeline import BasePipeline


class DynamicModel(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'dynamic_model')

    def run_test(self):
        hdf5 = glob.glob(
            '/apollo/modules/data/fuel/fueling/control/data/hdf5/*/*/*.hdf5')
        model_dirs = '/apollo/modules/data/fuel/fueling/control/data/model_output/'
        grading_dirs = '/apollo/modules/data/fuel/fueling/control/data/evaluation_result/'
        mlp_keras.mlp_keras(hdf5, model_dirs)
        files = glob.glob(
            '/apollo/modules/data/fuel/fueling/control/data/model_output/*.h5')
        h5s = glob.glob(
            '/apollo/modules/data/fuel/fueling/control/data/hdf5_evaluation/*.hdf5')
        self.model_evalution(files, h5s, model_dirs, grading_dirs)

    def run_prod(self):
        hdf5 = glob.glob(
            '/mnt/bos/modules/control/feature_extraction_hf5/hdf5_training/transit_2019/*/*/*.hdf5')
        model_dirs = '/mnt/bos/modules/control/dynamic_model_output'
        grading_dirs = '/mnt/bos/modules/control/evaluation_result'
        mlp_keras.mlp_keras(hdf5, model_dirs)
        files = glob.glob(
            "/mnt/bos/modules/control/dynamic_model_output/fnn_model_*.h5")
        h5s = glob.glob(
            '/mnt/bos/modules/control/feature_extraction_hf5/hdf5_evaluation/*.hdf5')
        self.model_evalution(files, h5s, model_dirs, grading_dirs)

    def load_model(self, files, sub_module):
        return (
            self.get_spark_context().parallelize(files)  # All the model files
            .filter(lambda x: sub_module in x)  # Model weights files
            .map(lambda x: self.extract_file_id(x, 'fnn_model_' + sub_module + '_', '.h5'))
            .distinct())

    def extract_file_id(self, file_name, start_position, end_position):
        model_id = file_name.split(start_position)[1].split(end_position)[0]
        return model_id

    def generate_segments(self, h5_file):
        segments = []
        print 'Loading {}'.format(h5_file)
        with h5py.File(h5_file, 'r+') as f:
            names = [n for n in f.keys()]
            if len(names) < 1:
                return
            for i in range(len(names)):
                data_segment = np.array(f[names[i]])
                segments.append(data_segment)
        print 'Segments count: ', len(segments)
        return segments

    def model_evalution(self, files, h5s, model_dirs, grading_dirs):
        print ("Files: %s" % files)
        model_weights = self.load_model(files, 'weights')
        model_norms = self.load_model(files, 'norms')

        records = (
            self.get_spark_context().parallelize(h5s)  # All the records for evaluation
            .map(lambda h5: (self.extract_file_id(h5, '/hdf5_evaluation/', '.hdf5'),
                             self.generate_segments(h5)))
            .cache())
        print('Get {} records'.format(records.count()))

        models = (
            model_weights
            .intersection(model_norms)
            .cartesian(records)
            .foreach(lambda pairs: trajectory_visualization.evaluate(pairs[0], pairs[1], model_dirs, grading_dirs)))


if __name__ == '__main__':
    DynamicModel().run_test()
