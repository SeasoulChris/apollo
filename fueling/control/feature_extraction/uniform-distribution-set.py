#!/usr/bin/env python
""" extracting even distributed sample set """
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.h5_utils as h5_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.features.feature_extraction_utils as feature_extraction_utils


# parameters
WANTED_VEHICLE = feature_extraction_utils.FEATURE_KEY.vehicle_type
counter = 0
sample_size = 100
def get_key(file_name):
    key, pre_segmentID = file_name.split('_')
    segmentID = os.path.splitext(pre_segmentID)[0]
    return key, segmentID


def pick_sample(list_of_segment):
    counter = 0
    sample_list = []
    for segment in list_of_segment:
        add_size = segment.shape[0]
        if counter + add_size < sample_size:
            counter += segment.shape[0]  # row, data points
            sample_list.append(segment)
            counter += add_size
        elif counter < sample_size:
            to_add_size = sample_size-counter+1
            sample_list.append(segment[0:to_add_size, :])
            return (sample_list, sample_size)
    return (sample_list, counter)


def write_to_file(target_prefix, elem):
    key, list_of_segment = elem
    total_number = len(list_of_segment)
    file_dir = os.path.join(target_prefix, key)
    counter = 1
    for data in list_of_segment:
        file_name = str(counter).zfill(4) + "_of_" + str(total_number).zfill(4)
        h5_utils.write_h5_single_segment(data, file_dir, file_name)
        counter += 1
    return total_number


class UniformDistributionSet(BasePipeline):
    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'uniform_distribution_set')

    def run_test(self):
        """Run test."""
        glog.info('WANTED_VEHICLE: %s' % WANTED_VEHICLE)
        origin_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'SampleSet')
        target_prefix = os.path.join('modules/data/fuel/testdata/control/generated',
                                     WANTED_VEHICLE, 'EvenlyDitributed')
        root_dir = '/apollo'
        files_dir = os.path.join(root_dir, origin_prefix)
        target_dir = os.path.join(root_dir, target_prefix)
        todo_tasks = (
            # RDD(all files)
            self.get_spark_context().parallelize(dir_utils.list_end_files(files_dir))
            #RDD(.hdf5 files)
            .filter(lambda path: path.endswith('.hdf5'))).cache()
        glog.info('NUMBER of TODO TASK: %d', todo_tasks.count())
        self.run(todo_tasks, target_dir)

    def run_prod(self):
        """Run prod."""
        bucket = 'apollo-platform'
        # same of target prefix of sample-set-feature-extraction
        origin_prefix = os.path.join('modules/control/feature_extraction_hf5/hdf5_training/',
                                     WANTED_VEHICLE, 'SampleSet')
        target_prefix = os.path.join('modules/control/feature_extraction_hf5/hdf5_training/',
                                     WANTED_VEHICLE, 'EvenlyDistributedSampleSet')
        root_dir = s3_utils.S3_MOUNT_PATH

        path = os.path.join(root_dir, origin_prefix)
        target_dir = os.path.join(root_dir, target_prefix)

        todo_tasks = (
            # RDD(file)
            s3_utils.list_files(bucket, path)
            # RDD(.hdf5 file)
            .filter(lambda path: path.endswith('.hdf5')))

        self.run(todo_tasks, target_dir)

    def run(self, todo_tasks, target_prefix):
        categorized_segments = (
            # RDD(.hdf5 files with absolute path)
            todo_tasks
            # PairRDD(file_path, file_name)
            .map(lambda file_dir: (file_dir, os.path.basename(file_dir)))
            # PairRDD(file_path, (key, segmentID))
            .mapValues(get_key)
            # PairRDD(key, file_path)
            .map(lambda elem: (elem[1][0], elem[0]))
            # PairRDD(key, segments)
            .mapValues(h5_utils.read_h5)
            # PairRDD(key, segments RDD)
            .groupByKey()
            # PairRDD(key, list of segments)
            .mapValues(list)
        )

        sampled_segments = (
            # PairRDD(key, list of segments)
            categorized_segments
            # PairRDD(key, (sampled segments, counter))
            .mapValues(pick_sample)
            # PairRDD(key, (sampled segments, counter=sample_size))
            .filter(lambda (_, segment_counter): segment_counter[1] == sample_size)
            # PairRDD(key, sampled segments)
            .mapValues(lambda segment_counter: segment_counter[0])
            # RDD(segment_length)
            .map(lambda elem: write_to_file(target_prefix, elem))
        ) 

        glog.info('Generated %d categories', sampled_segments.count())


if __name__ == '__main__':
    UniformDistributionSet().main()
