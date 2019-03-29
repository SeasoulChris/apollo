#!/usr/bin/env python
""" extracting even distributed sample set """
import os

import fueling.common.colored_glog as glog
from fueling.common.base_pipeline import BasePipeline
import fueling.common.h5_utils as h5_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils

# parameters
WANTED_VEHICLE = feature_extraction_utils.FEATURE_KEY.vehicle_type
training_size = 1 # configurable
counter = 0
sample_size = 10

def get_key(file_name):
    key, pre_segmentID = file_name.split('_')
    segmentID, file_type = pre_segmentID.split('.')
    return key, segmentID

def pick_sample(list_of_segment):
    counter = 0
    sample_list = []
    for i in range(len(list_of_segment)):
        add_size = list_of_segment[i].shape[0]
        if counter + add_size < sample_size:
            counter += list_of_segment[i].shape[0] # row, data points
            sample_list.append(list_of_segment[i])
        elif counter < sample_size:
            to_add_size = sample_size-counter+1
            sample_list.append(list_of_segment[i][0:to_add_size,:])
            return sample_list
    return sample_list

def write_to_file(target_prefix, elem):
    key, list_of_segment = elem
    total_number = len(list_of_segment)
    file_dir = os.path.join(target_prefix,key)
    print(file_dir)
    for i in range(total_number):
        data = list_of_segment[i]
        file_name = str(i+1).zfill(4) + "_of_" + str(total_number).zfill(4)
        print(file_name)
        h5_utils.write_h5_single_segment(data, file_dir, file_name)
    return total_number
            
class EvenDistributedFeatureExtraction(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'even_distributed_feature_extraction')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)
        origin_prefix = os.path.join('modules/data/fuel/testdata/control/generated', 
                                     WANTED_VEHICLE, 'SampleSet')
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'EvenlyDitributed')
        root_dir = '/apollo'
        files_dir = os.path.join(root_dir, origin_prefix)
        target_prefix = os.path.join(root_dir, target_prefix)
        glog.info(files_dir)
        todo_tasks = (
            #RDD(all files)
            self.get_spark_context().parallelize(dir_utils.list_end_files(files_dir))
            #RDD(.hdf5 files)
            .filter(lambda path: path.endswith('.hdf5')))
        glog.info(todo_tasks.first())
        self.run(todo_tasks, origin_prefix, target_prefix)
    
    def run_prod(self):
        """Run prod."""
    
    def run(self, todo_tasks, origin_prefix, target_prefix):
        categorized_segments = (
            # RDD(.hdf5 files with absolute path)
            todo_tasks
            # PairedRDD(file_path, file_name)
            .map(lambda file_dir: (file_dir, os.path.basename(file_dir)))
            # PairedRDD(file_path, key,segmentID)
            .mapValues(get_key)
            # PairedRDD(key, file_path)
            .map(lambda elem: (elem[1][0], elem[0]))
            # PairedRDD(key, segments)
            .mapValues(h5_utils.read_h5)
            # PairedRDD(key, list of segments)
            .combineByKey(feature_extraction_utils.to_list, feature_extraction_utils.append,
                          feature_extraction_utils.extend)
        )

        sampled_segments = (
            # PairedRDD(key, list of segments)
            categorized_segments
            # PairedRDD(key, sampled segments)
            .mapValues(pick_sample)
            # dRDD(segment_length)
            .map(lambda elem: write_to_file(target_prefix, elem))
        )

        testdata = sampled_segments.count()

if __name__ == '__main__':
    EvenDistributedFeatureExtraction().run_test()
