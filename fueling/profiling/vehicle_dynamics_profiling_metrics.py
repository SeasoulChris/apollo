#!/usr/bin/env python

""" Extracting features and grading the vehicle dynamic"""

from collections import namedtuple
import glob
import os

import colored_glog as glog
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.record_utils as record_utils
import fueling.profiling.common.dir_utils as dir_utils
import fueling.profiling.feature_extraction.vehicle_dynamics_feature_extraction_utils \
       as feature_utils
import fueling.profiling.grading_evaluation.vehicle_dynamics_grading_utils as grading_utils


class VehicleDynamicsProfilingMetrics(BasePipeline):

    def __init__(self):
        """Initialize """
        BasePipeline.__init__(self, 'vehicle dynamics profiling metrics')

    def run_test(self):
        """Run test."""

        original_prefix = '/apollo/modules/data/fuel/testdata/profiling/vehicle_dynamics'
        target_prefix = '/apollo/modules/data/fuel/testdata/profiling/vehicle_dynamics/generated'
        # RDD(tasks), the task dirs
        todo_tasks = self.to_rdd([
            os.path.join(original_prefix, 'Road_Test'),
        ]).cache()
        self.run(todo_tasks, original_prefix, target_prefix)
        # summarize_tasks(todo_tasks.collect(), original_prefix, target_prefix)
        glog.info('tasks {}'.format(todo_tasks.collect()))
        glog.info('Vehicle Dynamics Profiling: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = 'small-records/2019'
        target_prefix = 'modules/profiling/vehicle_dynamics_profiling_hf5'
        # RDD(tasks), the task dirs
        dir_todo_tasks = dir_utils.get_todo_tasks(original_prefix, target_prefix)
        todo_tasks = spark_helper.cache_and_log('todo_tasks', dir_todo_tasks)
        self.run(todo_tasks, original_prefix, target_prefix)
        # summarize_tasks(todo_tasks.collect(), original_prefix, target_prefix)
        glog.info('Control Profiling: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters, core procedure"""

        # RDD(tasks), with absolute paths
        (todo_tasks
         # RDD(tasks), filter the tasks that have configured values
         .filter(feature_utils.verify_vehicle_controller)
         # PairRDD(target_dir, task), the map of target dirs and source dirs
         .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1))
         # PairRDD(target_dir, record_file)
         .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')) +
                        glob.glob(os.path.join(task, '*bag*')))
         # PairRDD(target_dir, record_file), filter out unqualified files
         .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or
                                       record_utils.is_bag_file(file)))
         # PairRDD(target_dir, message), control and chassis message
         .flatMapValues(record_utils.read_record([record_utils.CONTROL_CHANNEL,
                                                  record_utils.CHASSIS_CHANNEL]))
         # PairRDD(target_dir, (message)s)
         .groupByKey()
         # RDD(target_dir, group_id, group of (message)s), divide messages into groups
         .flatMap(partition_data)
         # PairRDD(target_dir, matrix_result), for each group get the matrix and write h5 files
         .map(grading_utils.generating_matrix_and_h5)
         # PairRDD(target_dir, grading_result), for each group compute the gradings
         .map(grading_utils.computing_gradings)
         # PairRDD(target_dir, combined_grading_result), combine gradings for each target/task
         .reduceByKey(grading_utils.combine_gradings)
         # PairRDD(target_dir, combined_grading_result), output grading results for each target
         .foreach(grading_utils.output_gradings))


def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    glog.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msg: msg.timestamp)
    msgs_groups = [msgs[idx: idx + feature_utils.MSG_PER_SEGMENT]
                   for idx in range(0, len(msgs), feature_utils.MSG_PER_SEGMENT)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]


if __name__ == '__main__':
    VehicleDynamicsProfilingMetrics().main()
