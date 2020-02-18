#!/usr/bin/env python

import glob
import os

from absl import flags
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils


flags.DEFINE_string('learning_based_planning_input_path_local',
                    '/apollo/data/learning_based_planning',
                    'input data directory for local run_test')
flags.DEFINE_string('learning_based_planning_output_path_local',
                    '/apollo/data/learning_based_planning',
                    'output data directory for local run_test')


class EgoVehicleTrajectory(BasePipeline):

    def run_test(self):
        """ Run test. """
        # 1. get local record
        origin_prefix = flags.FLAGS.learning_based_planning_input_path_local
        target_prefix = flags.FLAGS.learning_based_planning_output_path_local
        # sub folders
        todo_tasks_dirs = [subdir for subdir in os.listdir(
            origin_prefix) if os.path.isdir(os.path.join(origin_prefix, subdir))]
        logging.info(F'todo_task_dirs: {todo_tasks_dirs}')
        # RDD(todo_task_dirs)
        todo_task_dirs = self.to_rdd([
            os.path.join(origin_prefix, task) for task in todo_tasks_dirs
        ]).cache()
        logging.info(F'todo_task_dirs: {todo_task_dirs.collect()}')
        # 2. run test
        self.run(todo_task_dirs, origin_prefix, target_prefix)

    def run(self, todo_task_dirs, origin_prefix, target_prefix):
        """ process records """
        # TODO(SHU):
        # 1. records to localization messages
        localization_msgs = (todo_task_dirs
                             # PairRDD(target_dir, task), the map of target dirs and source dirs
                             .keyBy(lambda source: source.replace(origin_prefix, target_prefix, 1))
                             # PairRDD(target_dir, record_file)
                             .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')) +
                                            glob.glob(os.path.join(task, '*bag*')))
                             # PairRDD(target_dir, record_file), filter out unqualified files
                             .filter(spark_op.filter_value(
                                 lambda file: record_utils.is_record_file(file) or
                                 record_utils.is_bag_file(file)))
                             # PairRDD(target_dir, message), localization message
                             .flatMapValues(record_utils.read_record([record_utils.LOCALIZATION_CHANNEL]))
                             # PairRDD(target_dir, parsed_message), parsed localization message
                             .mapValues(record_utils.message_to_proto))
        logging.info(F'localization_messeger_count: {localization_msgs.count()}')


if __name__ == '__main__':
    EgoVehicleTrajectory().main()
