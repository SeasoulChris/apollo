#!/usr/bin/env python

import glob
import os

from absl import flags

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging

flags.DEFINE_string('open_space_planner_profilling_input_path_local',
                    '/apollo/data/open_space_profiling',
                    'input data directory for local run_test')
flags.DEFINE_string('open_space_planner_profilling_output_path_local',
                    '/apollo/data/open_space_profiling_generated',
                    'output data directory for local run_test')


class OpenSpacePlannerMetrics(BasePipeline):

    def run_test(self):
        """ Run test. """
        # 1. get local record
        origin_prefix = flags.FLAGS.open_space_planner_profilling_input_path_local
        target_prefix = flags.FLAGS.open_space_planner_profilling_output_path_local
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

    def run(self, todo_tasks, original_prefix, target_prefix):
        """ process records """
        # TODO(SHU):
        # 1. records to planning messages
        # 2. filter messages belonging to a certain stage (stage name)
        # 3. get features from message (feature list)
        # 4. process feature (count, max, mean, standard deviation, 95 percentile)
        # 5. write result to target folder


if __name__ == '__main__':
    OpenSpacePlannerMetric().main()
