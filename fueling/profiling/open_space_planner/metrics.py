#!/usr/bin/env python

from collections import namedtuple
import glob
import os
import shutil
import tarfile
import time

from absl import flags
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
from fueling.profiling.open_space_planner.feature_extraction.feature_visualization_utils import \
    plot
from fueling.profiling.open_space_planner.feature_extraction.feature_extraction_utils import \
    extract_latency_feature, extract_planning_trajectory_feature, extract_stage_feature, \
    extract_zigzag_trajectory_feature, output_features
from fueling.profiling.open_space_planner.metrics_utils.evaluation_method_util import \
    latency_grading, merge_grading_results, output_grading, stage_grading, trajectory_grading, \
    zigzag_grading
from modules.planning.proto.planning_config_pb2 import ScenarioConfig

flags.DEFINE_string('open_space_planner_profiling_input_path',
                    '/apollo/data/open_space_profiling',
                    'input data directory')
flags.DEFINE_string('open_space_planner_profiling_output_path',
                    '/apollo/data/open_space_profiling_generated',
                    'output data directory')
flags.DEFINE_boolean('open_space_planner_profiling_generate_report', True,
                     'whether an email report with feature plot etc. is required')

SCENARIO_TYPE = ScenarioConfig.VALET_PARKING
STAGE_TYPE = ScenarioConfig.VALET_PARKING_PARKING
VEHICLE_PARAM_FILE = 'vehicle_param.pb.txt'


def zip_msgs(msgs):
    """convert a list of sorted prediction-and-planning msgs into a list of planning-prediction pairs"""
    paired_msgs = []
    latest_prediction_pb2 = None
    for msg in msgs:
        msg_pb2 = record_utils.message_to_proto(msg)
        if not hasattr(msg_pb2, 'header'):
            logging.warn(f"Found a message without header: {msg_pb2}")
            continue

        module_name = msg_pb2.header.module_name
        if module_name == 'planning':
            paired_msgs.append({
                'planning': msg_pb2,
                'prediction': latest_prediction_pb2
            })
        elif module_name == 'prediction':
            latest_prediction_pb2 = msg_pb2
        else:
            logging.error(f"Unknown msg found: {msg_pb2.header}")

    return paired_msgs


def has_desired_stage(msgs):
    if hasattr(msgs['planning'].debug.planning_data, 'scenario'):
        scenario = msgs['planning'].debug.planning_data.scenario
        return (scenario.scenario_type == SCENARIO_TYPE and
                scenario.stage_type == STAGE_TYPE)
    return False


def sort_messages(msgs):
    logging.info(F'Sorting {len(msgs)} messages')
    return sorted(msgs, key=lambda msg: msg['planning'].header.sequence_num)


def copy_if_needed(src, dst):
    if os.path.exists(dst):
        logging.info(F'No need to copy {dst}')
        return
    shutil.copyfile(src, dst)


class OpenSpacePlannerMetrics(BasePipeline):

    def run(self):
        tic = time.perf_counter()
        # 1. get record files
        origin_prefix = flags.FLAGS.open_space_planner_profiling_input_path
        target_prefix = flags.FLAGS.open_space_planner_profiling_output_path
        our_storage = self.our_storage()

        # Access partner's storage if provided.
        object_storage = self.partner_storage() or our_storage

        # This will return absolute path
        src_dirs = self.to_rdd(object_storage.list_end_dirs(origin_prefix))
        if logging.level_debug():
            logging.debug(F'src_dirs: {src_dirs.collect()}')

        # Copy over vehicle param config
        src_dst_rdd = (src_dirs
                       .map(lambda src_dir: (
                            src_dir, src_dir.replace(origin_prefix, target_prefix, 1)))
                       .cache())
        if logging.level_debug():
            logging.debug(F'src_dst_rdd: {src_dst_rdd.collect()}')
        src_dst_rdd.values().foreach(file_utils.makedirs)
        src_dst_rdd.foreach(
            lambda src_dst: copy_if_needed(
                os.path.join(src_dst[0], VEHICLE_PARAM_FILE),
                os.path.join(src_dst[1], VEHICLE_PARAM_FILE)))

        # PairRDD(todo_task_dirs)
        todo_task_dirs = (src_dirs
                          # PairRDD(target_prefix, src_dir), the map of target dirs and source dirs
                          .keyBy(lambda source: source.replace(origin_prefix, target_prefix, 1))
                          # PairRDD(target_prefix, file)
                          .flatMapValues(object_storage.list_files)
                          # PairRDD(target_prefix, record_file)
                          .filter(spark_op.filter_value(
                              lambda file: record_utils.is_record_file(file) or
                              record_utils.is_bag_file(file)))
                          .cache())
        if logging.level_debug():
            logging.debug(F'todo_task_dirs: {todo_task_dirs.collect()}')
        if not todo_task_dirs.collect():
            logging.info('No data to perform open space planner profilng on.')
            return
        logging.info(F'Preparing record files took {time.perf_counter() - tic:0.3f} sec')

        # 2. run evaluation
        return self.process(todo_task_dirs, origin_prefix, target_prefix)

    def process(self, todo_task_dirs, origin_prefix, target_prefix):
        """ process records """
        tic = time.perf_counter()
        msgs = (todo_task_dirs
                # PairRDD(target_prefix, raw_message)
                .mapValues(record_utils.read_record(
                    [record_utils.PLANNING_CHANNEL, record_utils.PREDICTION_CHANNEL]))
                # PairRDD(target_prefix, {planning_pb2, prediction_pb2})
                .flatMapValues(zip_msgs))
        if logging.level_debug():
            logging.debug(F'msg count: {msgs.count()}')
            logging.debug(F'msg first: {msgs.first()}')

        # 2. filter messages belonging to a certain stage (STAGE_TYPE)
        open_space_msgs = (msgs
                           # PairRDD(target_prefix, filtered_message),
                           # keep message with desired stage
                           .filter(spark_op.filter_value(has_desired_stage))
                           # PairRDD(target_prefix, iter[filtered_message])
                           .groupByKey()
                           # PairRDD(target_prefix, filtered_and_sorted_messages)
                           .mapValues(sort_messages)
                           .cache())
        if logging.level_debug():
            logging.debug(F'open_space_message_count: {open_space_msgs.count()}')
            logging.debug(F'open_space_message_first: {open_space_msgs.first()}')

        # 3. get feature from all frames with desired stage
        stage_feature = open_space_msgs.map(extract_stage_feature)
        latency_feature = open_space_msgs.map(extract_latency_feature)
        zigzag_feature = open_space_msgs.map(extract_zigzag_trajectory_feature)
        trajectory_feature = open_space_msgs.map(extract_planning_trajectory_feature)
        if logging.level_debug():
            logging.debug(F'stage_feature_count: {stage_feature.count()}')
            logging.debug(F'stage_feature_first: {stage_feature.first()}')
            logging.debug(F'latency_feature_count: {latency_feature.count()}')
            logging.debug(F'latency_feature_first: {latency_feature.first()}')
            logging.debug(F'zigzag_feature_count: {zigzag_feature.count()}')
            logging.debug(F'zigzag_feature_first: {zigzag_feature.first()}')
            logging.debug(F'trajectory_feature_count: {trajectory_feature.count()}')
            logging.debug(F'trajectory_feature_first: {trajectory_feature.first()}')

        # 4. grading, process feature (count, max, mean, standard deviation, 95 percentile)
        stage_result = stage_feature.map(stage_grading)
        latency_result = latency_feature.map(latency_grading)
        zigzag_result = zigzag_feature.map(zigzag_grading)
        trajectory_result = trajectory_feature.map(trajectory_grading)
        grading_result = (stage_result
                          .join(latency_result)
                          .mapValues(merge_grading_results)
                          .join(zigzag_result)
                          .mapValues(merge_grading_results)
                          .join(trajectory_result)
                          .mapValues(merge_grading_results))
        if logging.level_debug():
            logging.debug(F'grading_result_count: {grading_result.count()}')
            logging.debug(F'grading_result_first: {grading_result.first()}')

        # 5. plot and visualize features, save grading result
        if self.FLAGS['open_space_planner_profiling_generate_report']:
            # PairRDD(target, features), save features in h5 file
            stage_feature.foreach(lambda group: output_features(group, 'stage_feature'))
            latency_feature.foreach(lambda group: output_features(group, 'latency_feature'))
            zigzag_feature.foreach(lambda group: output_features(group, 'zigzag_feature'))
            trajectory_feature.foreach(lambda group: output_features(group, 'trajectory_feature'))
            # PairRDD(target, combined_grading_result), output grading results for each target
            grading_result.foreach(output_grading)
            # PairRDD(target, planning_trajectory_features), feature plots
            trajectory_feature.foreach(plot)
            self.email_output(todo_task_dirs.keys().collect(), origin_prefix, target_prefix)
            logging.info(F'Evaluation with report took {time.perf_counter() - tic:0.3f} sec')
        else:
            results = grading_result.collect()
            logging.info(F'Evaluation alone took {time.perf_counter() - tic:0.3f} sec')
            return results

    def email_output(self, tasks, origin_prefix, target_prefix, partner_email='', error_msg=''):
        title = 'Open Space Planner Profiling Results'
        recipients = email_utils.CONTROL_TEAM + email_utils.SIMULATION_TEAM
        # Uncomment this for dev test
        # recipients = ['caoyu05@baidu.com']
        recipients.append(partner_email)
        SummaryTuple = namedtuple('Summary', ['Task', 'FeatureHDF5s', 'FeaturePlot', 'Profiling'])
        if tasks:
            email_content = []
            attachments = []
            tar_filename = None
            tar = None
            tasks.sort()
            for task in tasks:
                source = task.replace(target_prefix, origin_prefix, 1)
                logging.info(F'task: {task}, source: {source}')
                features = glob.glob(os.path.join(task, '*.hdf5'))
                plot = glob.glob(os.path.join(task, '*visualization*'))
                profiling = glob.glob(os.path.join(task, '*performance_grading*'))
                email_content.append(SummaryTuple(
                    Task=source,
                    FeatureHDF5s=len(features),
                    FeaturePlot=len(plot),
                    Profiling=len(profiling),
                ))
                if profiling or features or plot:
                    tar_filename = F'{os.path.basename(source)}_profiling.tar.gz'
                    tar = tarfile.open(tar_filename, 'w:gz')
                    for report in profiling:
                        tar.add(report)
                    for report in features:
                        tar.add(report)
                    for report in plot:
                        tar.add(report)
                    tar.close()
            attachments.append(tar_filename)
        else:
            logging.info('todo_task_dirs: None')
            if error_msg:
                email_content = error_msg
            else:
                email_content = 'No profiling results: No raw data.'
            attachments = []

        email_utils.send_email_info(
            title, email_content, recipients, attachments)


if __name__ == '__main__':
    OpenSpacePlannerMetrics().main()
