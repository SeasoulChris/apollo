#!/usr/bin/env python

import glob
import os

from absl import flags
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils

from modules.planning.proto.planning_config_pb2 import ScenarioConfig

flags.DEFINE_string('open_space_planner_profilling_input_path_local',
                    '/apollo/data/open_space_profiling',
                    'input data directory for local run_test')
flags.DEFINE_string('open_space_planner_profilling_output_path_local',
                    '/apollo/data/open_space_profiling_generated',
                    'output data directory for local run_test')
SCENARIO_TYPE = ScenarioConfig.VALET_PARKING
STAGE_TYPE = ScenarioConfig.VALET_PARKING_PARKING


def has_scenario_info(parsed_planning_msg):
    return hasattr(getattr(getattr(parsed_planning_msg, 'debug'), 'planning_data'), 'scenario')


def is_right_stage(parsed_planning_msg):
    scenario = getattr(getattr(getattr(parsed_planning_msg, 'debug'), 'planning_data'), 'scenario')
    return (scenario.scenario_type == SCENARIO_TYPE and
            scenario.stage_type == STAGE_TYPE)


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
        self.run(todo_task_dirs, origin_prefix, target_prefix)

    def run(self, todo_task_dirs, origin_prefix, target_prefix):
        """ process records """
        # TODO(SHU):
        # 1. records to planning messages
        planning_msgs = (todo_task_dirs
                         # PairRDD(target_dir, task), the map of target dirs and source dirs
                         .keyBy(lambda source: source.replace(origin_prefix, target_prefix, 1))
                         # PairRDD(target_dir, record_file)
                         .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')) +
                                        glob.glob(os.path.join(task, '*bag*')))
                         # PairRDD(target_dir, record_file), filter out unqualified files
                         .filter(spark_op.filter_value(
                             lambda file: record_utils.is_record_file(file) or
                             record_utils.is_bag_file(file)))
                         # PairRDD(target_dir, message), planing message
                         .flatMapValues(record_utils.read_record([record_utils.PLANNING_CHANNEL]))
                         # PairRDD(target_dir, parsed_message), parsed planing message
                         .mapValues(record_utils.message_to_proto)
                         # PairRDD(target_dir, parsed_message), parsed message with scenario info
                         .filter(spark_op.filter_value(has_scenario_info))
                         # PairRDD(target_dir, parsed_message), parsed message in desired stage
                         .filter(spark_op.filter_value(is_right_stage)))
        logging.info(F'planning_messeger_count: {planning_msgs.count()}')
        # logging.info(F'planning_messeger_first: {planning_msgs.first()}')

        # 2. filter messages belonging to a certain stage (stage name)
        # 3. get features from message (feature list)
        # 4. process feature (count, max, mean, standard deviation, 95 percentile)
        # 5. write result to target folder


if __name__ == '__main__':
    OpenSpacePlannerMetric().main()
