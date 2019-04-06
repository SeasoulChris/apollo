#!/usr/bin/env python

"""Extracting features and grading the control performance based on the designed metrics"""

import glob
import os

import colored_glog as glog
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.record_utils as record_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.features.dir_utils as dir_utils
import fueling.control.control_profiling.feature_extraction.control_feature_extraction_utils \
       as feature_utils 
import fueling.control.control_profiling.grading_evaluation.control_performance_grading_utils \
       as grading_utils


class ControlProfilingMetrics(BasePipeline):
    """ Control Profiling: Feature Extraction and Performance Grading """

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'control_profiling_metrics')

    def run_test(self):
        """Run test."""
        origin_prefix = 'modules/data/fuel/testdata/control/control_profiling'
        target_prefix = 'modules/data/fuel/testdata/control/control_profiling/generated'
        root_dir = '/apollo'
        # RDD(tasks), the task dirs
        todo_tasks = self.get_spark_context().parallelize([
            os.path.join(origin_prefix, 'Transit_Auto'),
            os.path.join(origin_prefix, 'Transit_Auto2')
        ])
        self.run(todo_tasks, root_dir, origin_prefix, target_prefix)
        glog.info('Control Profiling: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = 'small-records/2019'
        target_prefix = 'modules/control/control_profiling_hf5'
        bucket = 'apollo-platform'
        # RDD(tasks), the task dirs
        todo_tasks = dir_utils.get_todo_tasks(original_prefix,
                                              target_prefix,
                                              lambda path: s3_utils.list_files(bucket, path))
        glog.info('todo tasks: {}'.format(todo_tasks.collect()))
        self.run(todo_tasks, s3_utils.S3_MOUNT_PATH, original_prefix, target_prefix)
        glog.info('Control Profiling: All Done, PROD')

    def run(self, todo_tasks, root_dir, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
        # RDD(tasks), with relative paths
        (todo_tasks
         # RDD(tasks), with absolute paths
         .map(lambda task: os.path.join(root_dir, task))
         # RDD(tasks), filter the tasks that have configured values
         .filter(feature_utils.verify_vehicle_controller)
         # PairRDD(target_dir, task), the map of target dirs and source dirs
         .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1))
         # PairRDD(target_dir, record_file)
         .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')))
         # PairRDD(target_dir, record_file), filter out unqualified files
         .filter(spark_op.filter_value(record_utils.is_record_file))
         # PairRDD(target_dir, message), control message only
         .flatMapValues(record_utils.read_record([record_utils.CONTROL_CHANNEL]))
         # PairRDD(target_dir, (message)s)
         .groupByKey()
         # PairRDD(target_dir, (message)s), divide messages into groups
         .flatMap(partition_data)
         # PairRDD(target_dir, grading_result), for each group get the gradings and write h5 files
         .map(grading_utils.compute_h5_and_gradings)
         # PairRDD(target_dir, combined_grading_result), combine gradings for each target/task
         .reduceByKey(grading_utils.combine_gradings)
         # PairRDD(target_dir, combined_grading_result), output grading results for each target
         .map(grading_utils.output_gradings)
         # Trigger actions
         .count())

def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    glog.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msg: msg.timestamp)
    msgs_groups = [msgs[idx: idx + grading_utils.MSG_PER_SEGMENT]
                   for idx in range(0, len(msgs), grading_utils.MSG_PER_SEGMENT)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]

if __name__ == '__main__':
    ControlProfilingMetrics().main()
