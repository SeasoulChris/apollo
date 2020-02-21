#!/usr/bin/env python

import glob
import os

from absl import flags
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
# TODO(SHU): will refract it later
from fueling.profiling.open_space_planner.feature_extraction.feature_extraction_utils import extract_mtx
from fueling.profiling.open_space_planner.feature_extraction.feature_extraction_utils import extract_mtx_repeated_field
from fueling.profiling.open_space_planner.metrics_utils.evaluation_method_util import grading, output_result
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.profiling.control.feature_visualization.control_feature_visualization_utils \
    as visual_utils

from modules.planning.proto.planning_config_pb2 import ScenarioConfig

flags.DEFINE_string('open_space_planner_profilling_input_path_local',
                    '/apollo/data/open_space_profiling',
                    'input data directory for local run_test')
flags.DEFINE_string('open_space_planner_profilling_output_path_local',
                    '/apollo/data/open_space_profiling_generated',
                    'output data directory for local run_test')
SCENARIO_TYPE = ScenarioConfig.VALET_PARKING
STAGE_TYPE = ScenarioConfig.VALET_PARKING_PARKING
MSG_PER_SEGMENT = 100000


def has_scenario_info(parsed_planning_msg):
    return hasattr(parsed_planning_msg.debug.planning_data, 'scenario')


def is_right_stage(parsed_planning_msg):
    scenario = parsed_planning_msg.debug.planning_data.scenario
    return (scenario.scenario_type == SCENARIO_TYPE and
            scenario.stage_type == STAGE_TYPE)


def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    logging.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msg: msg.header.sequence_num)
    msgs_groups = [msgs[idx: idx + MSG_PER_SEGMENT]
                   for idx in range(0, len(msgs), MSG_PER_SEGMENT)]
    # return msgs

    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]


class OpenSpacePlannerMetrics(BasePipeline):

    def run(self):
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
        self.run_internal(todo_task_dirs, origin_prefix, target_prefix)

    def run_internal(self, todo_task_dirs, origin_prefix, target_prefix):
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
                         .mapValues(record_utils.message_to_proto))
        logging.info(F'planning_messeger_count: {planning_msgs.count()}')

        # 2. filter messages belonging to a certain stage (stage name)
        open_space_msgs = (planning_msgs
                           # PairRDD(target_dir, parsed_message), parsed message with scenario info
                           .filter(spark_op.filter_value(has_scenario_info))
                           # PairRDD(target_dir, parsed_message), parsed message in desired stage
                           .filter(spark_op.filter_value(is_right_stage))
                           ).cache()
        logging.info(F'open_space_messeger_count: {open_space_msgs.count()}')
        # logging.info(F'feature_data_first: {open_space_msgs.first()}')

        # # 3. get features from message (feature list)
        # feature_data = (open_space_msgs
        #                 # PairRDD(target_dir, parsed_message), parsed message with scenario info
        #                 .groupByKey()
        #                 # RDD(target_dir, group_id, group of (message)s), divide messages into groups
        #                 .flatMap(partition_data)
        #                 .map(extract_mtx))
        # logging.info(F'feature_data_count: {feature_data.count()}')
        # logging.info(F'feature_data_first: {feature_data.first()}')

        # # 4. process feature (count, max, mean, standard deviation, 95 percentile)
        # result_data = (feature_data
        #                .map(grading))
        # logging.info(F'result_data_count: {result_data.count()}')
        # logging.info(F'result_data_first: {result_data.first()}')

        # get feature from all frames in designated stage
        feature_data = (open_space_msgs
                        # PairRDD(target_dir, parsed_message), parsed message with scenario info
                        .groupByKey()
                        # RDD(target_dir, group_id, group of (message)s), divide messages into
                        # groups
                        .flatMap(partition_data)
                        .map(extract_mtx_repeated_field))
        logging.info(F'feature_data_count: {feature_data.count()}')
        logging.info(F'feature_data_first: {feature_data.first()}')

        result_data = (feature_data
                       .map(grading))
        logging.info(F'result_data_count: {result_data.count()}')
        logging.info(F'result_data_first: {result_data.first()}')

        # TODO(SHU): evaluation for all frames in a stage

        # 5. write result to target folder

        # plot
        feature_data.foreach(visual_utils.plot_hist)
        (result_data
         # PairRDD(target_dir, combined_grading_result), output grading results for each target
         .foreach(output_result))


if __name__ == '__main__':
    OpenSpacePlannerMetrics().main()
