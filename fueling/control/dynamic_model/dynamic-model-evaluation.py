#!/usr/bin/env python

import glob

import h5py
import numpy as np

import fueling.control.offline_evaluator.trajectory_visualization as trajectory_visualization

from fueling.common.base_pipeline import BasePipeline


class DynamicModelEvaluation(BasePipeline):
    def __init__(self):
        BasePipeline.__init__(self, 'dynamic_model')

    def run_test(self):
        dirs = '/apollo/modules/data/fuel/fueling/control/data/'
        files = glob.glob(
            '/apollo/modules/data/fuel/fueling/control/data/dynamic_model_output/*.h5')
        h5s = glob.glob('/apollo/modules/data/fuel/fueling/control/data/hdf5_evaluation/*.hdf5')
        self.model_evalution(files, h5s, dirs)

    def run_prod(self):
        dirs = '/mnt/bos/modules/control/'
        files = glob.glob('/mnt/bos/modules/control/dynamic_model_output/*_model_*.h5')
        h5s = glob.glob('/mnt/bos/modules/control/feature_extraction_hf5/hdf5_evaluation/*.hdf5')
        self.model_evalution(files, h5s, dirs)

    def load_model(self, files, sub_module):
        return (
            self.get_spark_context().parallelize(files)  # All the model files
            .filter(lambda x: sub_module in x)  # Model weights files
            .map(lambda x: (self.extract_file_id(x, 'dynamic_model_output/', '_model_' + sub_module + '_'),
                            self.extract_file_id(x, '_model_' + sub_module + '_', '.h5')))
            .distinct())

    def extract_file_id(self, file_name, start_position, end_position):
        return file_name.split(start_position)[1].split(end_position)[0]

    def generate_segments(self, h5_file):
        segments = []
        print 'Loading {}'.format(h5_file)
        with h5py.File(h5_file, 'r+') as fin:
            segments = [np.array(segment) for segment in fin.values()]
        print 'Segments count: ', len(segments)
        return segments

    def model_evalution(self, files, h5s, dirs):
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
            .foreach(lambda pairs: trajectory_visualization.evaluate(pairs[0], pairs[1], dirs)))


if __name__ == '__main__':
    DynamicModelEvaluation().run_prod()
