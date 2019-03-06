#!/usr/bin/env python
import sys
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
        self.model_training()
        self.model_evalution()

    def run_prod(self):
        return self.run_test()

    def load_model(self, files, sub_module):
        models = (spark_helper.get_context('Test')
                    .parallelize(files)  #all the model files
                    .filter(lambda x: sub_module in x) #model weights files
                    .map(lambda x: self.extract_file_id(x, 'fnn_model_' + sub_module +'_', '.h5'))
                    .distinct())
        return models

    def extract_file_id(self, file_name, start_position, end_position):
        model_id = file_name.split(start_position)[1].split(end_position)[0]
        return model_id

    def generate_segments(self, h5):
        segments = []
        print('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as f:
            names = [n for n in f.keys()]
            if len(names) < 1:
                return
            for i in range(len(names)):
                ds = np.array(f[names[i]])
                segments.append(ds)
        print('Segments count: ', len(segments))
        return segments

    def model_training(self):
        mlp_keras.mlp_keras('mlp_two_layer')

    def model_evalution(self):
        files = glob.glob("/mnt/bos/modules/control/dynamic_model_output/fnn_model_*.h5") #bos dirs
        #files = glob.glob('fueling/control/data/model_output/*.h5') #local dirs
        print ("Files: %s" % files)
        model_weights = self.load_model(files, 'weights')
        model_norms = self.load_model(files, 'norms')

        h5s = glob.glob('/mnt/bos/modules/control/feature_extraction_hf5/hdf5_evaluation/*.hdf5') #bos dirs
        #h5s = glob.glob('fueling/control/data/hdf5_evaluation/*.hdf5') #local dirs
        records = (spark_helper.get_context('Test')
                    .parallelize(h5s)  #all the records for evaluation
                    .map(lambda h5:(self.extract_file_id(h5, '/hdf5_evaluation/', '.hdf5'), self.generate_segments(h5)))
                    .cache())
        print('Get {} records'.format(records.count()))

        models = (model_weights
                    .intersection(model_norms)
                    .cartesian(records)
                    .foreach(lambda pairs: trajectory_visualization.evaluate(pairs[0], pairs[1])))

if __name__ == '__main__':
    # Gather result to memory and print nicely.
    DynamicModel().run_test()
