#!/usr/bin/env python

import glob
import os
import shutil

from absl import flags
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.profiling.control.feature_visualization.control_feature_visualization_utils \
    as visual_utils
from fueling.profiling.open_space_planner.feature_extraction.feature_extraction_utils import \
    extract_latency_feature, extract_planning_trajectory_feature, extract_zigzag_trajectory_feature
from fueling.profiling.open_space_planner.metrics_utils.evaluation_method_util import \
    latency_grading, merge_grading_results, output_result, trajectory_grading, zigzag_grading
from modules.planning.proto.planning_config_pb2 import ScenarioConfig

flags.DEFINE_string('open_space_planner_profiling_input_path',
                    '/apollo/data/open_space_profiling',
                    'input data directory')
flags.DEFINE_string('open_space_planner_profiling_output_path',
                    '/apollo/data/open_space_profiling_generated',
                    'output data directory')

SCENARIO_TYPE = ScenarioConfig.VALET_PARKING
STAGE_TYPE = ScenarioConfig.VALET_PARKING_PARKING
MSG_PER_SEGMENT = 100000
VEHICLE_PARAM_FILE = 'vehicle_param.pb.txt'


def has_desired_stage(parsed_planning_msg):
    if hasattr(parsed_planning_msg.debug.planning_data, 'scenario'):
        scenario = parsed_planning_msg.debug.planning_data.scenario
        return (scenario.scenario_type == SCENARIO_TYPE and
                scenario.stage_type == STAGE_TYPE)
    return False


def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    logging.info(F'partition data for {len(msgs)} messages in target {target}')
    msgs = sorted(msgs, key=lambda msg: msg.header.sequence_num)
    msgs_groups = [msgs[idx: idx + MSG_PER_SEGMENT]
                   for idx in range(0, len(msgs), MSG_PER_SEGMENT)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]


class OpenSpacePlannerMetrics(BasePipeline):

    def run(self):
        # 1. get record files
        origin_prefix = flags.FLAGS.open_space_planner_profiling_input_path
        target_prefix = flags.FLAGS.open_space_planner_profiling_output_path
        our_storage = self.our_storage()

        # Access partner's storage if provided.
        object_storage = self.partner_storage() or our_storage
        origin_dir = object_storage.abs_path(origin_prefix)
        logging.info(F'origin_dir: {origin_dir}')

        target_dir = our_storage.abs_path(target_prefix)
        logging.info(F'target_dir: {target_dir}')

        src_dirs = self.to_rdd(object_storage.list_end_dirs(origin_dir))

        # Copy over vehicle param config
        src_dst_rdd = (src_dirs
                       .map(lambda src_dir: (src_dir, src_dir.replace(origin_dir, target_dir, 1)))
                       .cache())
        logging.info(F'src_dst_rdd: {src_dst_rdd.collect()}')
        src_dst_rdd.values().foreach(file_utils.makedirs)
        src_dst_rdd.foreach(
            lambda src_dst: shutil.copyfile(
                os.path.join(src_dst[0], VEHICLE_PARAM_FILE),
                os.path.join(src_dst[1], VEHICLE_PARAM_FILE)))

        # PairRDD(todo_task_dirs)
        todo_task_dirs = (src_dirs
                          # PairRDD(target_dir, source_dir), the map of target dirs and source dirs
                          .keyBy(lambda source: source.replace(origin_dir, target_dir, 1))
                          # PairRDD(target_dir, file)
                          .flatMapValues(object_storage.list_files)
                          # PairRDD(target_dir, record_file)
                          .filter(spark_op.filter_value(
                              lambda file: record_utils.is_record_file(file) or
                              record_utils.is_bag_file(file)))
                          .cache())
        logging.info(F'todo_task_dirs: {todo_task_dirs.collect()}')

        # 2. run evaluation
        self.process(todo_task_dirs, origin_prefix, target_prefix)

    def process(self, todo_task_dirs, origin_prefix, target_prefix):
        """ process records """
        # 1. records to planning messages
        planning_msgs = (todo_task_dirs
                         # PairRDD(target_dir, message), planing message
                         .flatMapValues(record_utils.read_record([record_utils.PLANNING_CHANNEL]))
                         # PairRDD(target_dir, parsed_message), parsed planing message
                         .mapValues(record_utils.message_to_proto))
        logging.info(F'planning_messeger_count: {planning_msgs.count()}')

        # 2. filter messages belonging to a certain stage (STAGE_TYPE)
        open_space_msgs = (planning_msgs
                           # PairRDD(target_dir, parsed_message), keep message with desired stage
                           .filter(spark_op.filter_value(has_desired_stage))
                           ).cache()
        logging.info(F'open_space_messeger_count: {open_space_msgs.count()}')
        # logging.info(F'open_space_msgs_first: {open_space_msgs.first()}')

        # 3. get feature from all frames with desired stage
        raw_data = (open_space_msgs
                    # PairRDD(target_dir, filtered_message)
                    .groupByKey()
                    # RDD(target_dir, group_id, group of (message)s),
                    # divide messages into groups
                    .flatMap(partition_data))
        latency_feature = raw_data.map(extract_latency_feature)
        logging.info(F'latency_feature_count: {latency_feature.count()}')
        logging.info(F'latency_feature_first: {latency_feature.first()}')
        zigzag_feature = raw_data.map(extract_zigzag_trajectory_feature)
        logging.info(F'zigzag_feature_count: {zigzag_feature.count()}')
        logging.info(F'zigzag_feature_first: {zigzag_feature.first()}')
        trajectory_feature = raw_data.map(extract_planning_trajectory_feature)
        logging.info(F'trajectory_feature_count: {trajectory_feature.count()}')
        logging.info(F'trajectory_feature_first: {trajectory_feature.first()}')

        # 4. grading, process feature (count, max, mean, standard deviation, 95 percentile)
        latency_result = latency_feature.map(latency_grading)
        zigzag_result = zigzag_feature.map(zigzag_grading)
        trajectory_result = trajectory_feature.map(trajectory_grading)
        result_data = (latency_result
                       .join(zigzag_result)
                       .mapValues(merge_grading_results)
                       .join(trajectory_result)
                       .mapValues(merge_grading_results))
        logging.info(F'result_data_count: {result_data.count()}')
        logging.info(F'result_data_first: {result_data.first()}')

        # 5. plot and visualize features, save grading result
        trajectory_feature.foreach(visual_utils.plot_hist)
        (result_data
         # PairRDD(target_dir, combined_grading_result), output grading results for each target
         .foreach(output_result))


if __name__ == '__main__':
    OpenSpacePlannerMetrics().main()
