#!/usr/bin/env python
import sys
import glob
import h5py
import numpy as np
import fueling.common.spark_utils as spark_utils
import fueling.control.offline_evaluator.trajectory_visualization as trajectory_visualization

def extract_model(files, sub_module):
    models = (spark_utils.GetContext('Test')
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

def Main():
    files = glob.glob("/mnt/bos/modules/control/dynamic_model_output/fnn_model_*.h5")
    print ("Files: %s" % files)
    model_weights = extract_model(files, 'weights')
    model_norms = extract_model(files, 'norms')

    h5s = glob.glob('/mnt/bos/modules/control/feature_extraction_hf5/hdf5_evaluation/*.hdf5')
    records = (spark_utils.GetContext('Test')
                .parallelize(h5s)  #all the records for evaluation
                .map(lambda h5:(extract_file_id(h5, '/hdf5_evaluation/', '.hdf5'), generate_segments(h5)))
                .cache())
    print('Get {} records'.format(records.count()))

    models = (model_weights
                .intersection(model_norms)
                .cartesian(records)
                .foreach(lambda pairs: trajectory_visualization.evaluate(pairs[0], pairs[1], 'fueling/control/')))

if __name__ == '__main__':
    Main()
