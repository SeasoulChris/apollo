#!/usr/bin/env python
import sys
import glob

import h5py
import numpy as np
import pyspark_utils.helper as spark_helper

import fueling.control.offline_evaluator.trajectory_visualization as trajectory_visualization
import fueling.control.training_models.mlp_keras as mlp_keras


def load_model(files, sub_module):
    models = (spark_helper.get_context('Test')
                .parallelize(files)  #all the model files
                .filter(lambda x: sub_module in x) #model weights files
                .map(lambda x: extract_file_id(x, 'fnn_model_' + sub_module +'_', '.h5'))
                .distinct())
    return models

def extract_file_id(file_name, start_position, end_position):
    model_id = file_name.split(start_position)[1].split(end_position)[0]
    return model_id

def generate_segments(h5):
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

def model_evalution():
    files = glob.glob("/mnt/bos/modules/control/dynamic_model_output/fnn_model_*.h5")
    print ("Files: %s" % files)
    model_weights = load_model(files, 'weights')
    model_norms = load_model(files, 'norms')

    h5s = glob.glob('/mnt/bos/modules/control/feature_extraction_hf5/hdf5_evaluation/*.hdf5')
    records = (spark_helper.get_context('Test')
                .parallelize(h5s)  #all the records for evaluation
                .map(lambda h5:(extract_file_id(h5, '/hdf5_evaluation/', '.hdf5'), generate_segments(h5)))
                .cache())
    print('Get {} records'.format(records.count()))

    models = (model_weights
                .intersection(model_norms)
                .cartesian(records)
                .foreach(lambda pairs: trajectory_visualization.evaluate(pairs[0], pairs[1])))

def model_training():
    mlp_keras.mlp_keras('mlp_two_layer')

def Main():
    model_training()
    model_evalution()

if __name__ == '__main__':
    Main()
