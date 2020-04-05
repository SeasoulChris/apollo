#!/usr/bin/env python

from collections import namedtuple
import glob
import os
import shutil
import tarfile

from absl import flags
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.profiling.control.feature_visualization.control_feature_visualization_utils \
    as visual_utils
from fueling.profiling.open_space_planner.feature_extraction.feature_extraction_utils import \
    extract_latency_feature, extract_planning_trajectory_feature, extract_stage_feature, \
    extract_zigzag_trajectory_feature
from fueling.profiling.open_space_planner.metrics_utils.evaluation_method_util import \
    latency_grading, merge_grading_results, output_result, stage_grading, trajectory_grading, \
    zigzag_grading
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


def combine_msg_keys(rdds):
    planning_pair_rdd, prediction_pair_rdd = rdds
    planning_key, planning_rdd = planning_pair_rdd
    prediction_key, prediction_rdd = prediction_pair_rdd
    if planning_key != prediction_key:
        raise Exception(F"Wrong key {planning_key} vs. {prediction_key}")
    return (planning_key, {'planning': planning_rdd, 'prediction': prediction_rdd})


def has_desired_stage(msgs):
    if hasattr(msgs['planning'].debug.planning_data, 'scenario'):
        scenario = msgs['planning'].debug.planning_data.scenario
        return (scenario.scenario_type == SCENARIO_TYPE and
                scenario.stage_type == STAGE_TYPE)
    return False


def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    logging.info(F'Partition data for {len(msgs)} messages in target {target}')

    msgs = sorted(msgs, key=lambda msg: msg['planning'].header.sequence_num)
    msgs_groups = [msgs[idx: idx + MSG_PER_SEGMENT]
                   for idx in range(0, len(msgs), MSG_PER_SEGMENT)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]


def read_channel_messages(todo_task_dirs, channel):
    channel_msgs = (todo_task_dirs
                    # PairRDD(target_prefix, message)
                    .flatMapValues(record_utils.read_record([channel]))
                    # PairRDD(target_prefix, parsed_message)
                    .mapValues(record_utils.message_to_proto))
    logging.info(F'{channel} count: {channel_msgs.count()}')
    logging.debug(F'{channel} first: {channel_msgs.first()}')
    return channel_msgs


def copyIfNeeded(src, dst):
    if os.path.exists(dst):
        logging.info(F'No need to copy {dst}')
        return
    shutil.copyfile(src, dst)


class OpenSpacePlannerMetrics(BasePipeline):

    def run(self):
        # 1. get record files
        origin_prefix = flags.FLAGS.open_space_planner_profiling_input_path
        target_prefix = flags.FLAGS.open_space_planner_profiling_output_path
        our_storage = self.our_storage()

        # Access partner's storage if provided.
        object_storage = self.partner_storage() or our_storage

        # This will return absolute path
        src_dirs = self.to_rdd(object_storage.list_end_dirs(origin_prefix))
        logging.info(F'src_dirs: {src_dirs.collect()}')

        # Copy over vehicle param config
        src_dst_rdd = (src_dirs
                       .map(lambda src_dir: (
                            src_dir, src_dir.replace(origin_prefix, target_prefix, 1)))
                       .cache())
        logging.info(F'src_dst_rdd: {src_dst_rdd.collect()}')
        src_dst_rdd.values().foreach(file_utils.makedirs)
        src_dst_rdd.foreach(
            lambda src_dst: copyIfNeeded(
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
        logging.info(F'todo_task_dirs: {todo_task_dirs.collect()}')

        # 2. run evaluation
        self.process(todo_task_dirs, origin_prefix, target_prefix)

    def process(self, todo_task_dirs, origin_prefix, target_prefix):
        """ process records """
        planning_msgs = read_channel_messages(todo_task_dirs, record_utils.PLANNING_CHANNEL)
        prediction_msgs = read_channel_messages(todo_task_dirs, record_utils.PREDICTION_CHANNEL)
        if planning_msgs.count() != prediction_msgs.count():
            raise Exception(F'Different number of messages found between planning and predcition.')

        msgs = (planning_msgs
                # RDD((target_prefix, planning), (target_prefix, prediction))
                .zip(prediction_msgs)
                # PairRDD(target_prefix, {planning, prediction})
                .map(combine_msg_keys)
                )

        # 2. filter messages belonging to a certain stage (STAGE_TYPE)
        open_space_msgs = (msgs
                           # PairRDD(target_prefix, parsed_message), keep message with desired stage
                           .filter(spark_op.filter_value(has_desired_stage))
                           ).cache()
        logging.info(F'open_space_message_count: {open_space_msgs.count()}')
        logging.debug(F'open_space_message_first: {open_space_msgs.first()}')

        # 3. get feature from all frames with desired stage
        raw_data = (open_space_msgs
                    # PairRDD(target_prefix, filtered_messages)
                    .groupByKey()
                    # RDD(target_prefix, group_id, group of (message)s),
                    # divide messages into groups
                    .flatMap(partition_data))
        stage_feature = raw_data.map(extract_stage_feature)
        logging.info(F'stage_feature_count: {stage_feature.count()}')
        logging.debug(F'stage_feature_first: {stage_feature.first()}')
        latency_feature = raw_data.map(extract_latency_feature)
        logging.info(F'latency_feature_count: {latency_feature.count()}')
        logging.debug(F'latency_feature_first: {latency_feature.first()}')
        zigzag_feature = raw_data.map(extract_zigzag_trajectory_feature)
        logging.info(F'zigzag_feature_count: {zigzag_feature.count()}')
        logging.debug(F'zigzag_feature_first: {zigzag_feature.first()}')
        trajectory_feature = raw_data.map(extract_planning_trajectory_feature)
        logging.info(F'trajectory_feature_count: {trajectory_feature.count()}')
        logging.debug(F'trajectory_feature_first: {trajectory_feature.first()}')

        # 4. grading, process feature (count, max, mean, standard deviation, 95 percentile)
        stage_result = stage_feature.map(stage_grading)
        latency_result = latency_feature.map(latency_grading)
        zigzag_result = zigzag_feature.map(zigzag_grading)
        trajectory_result = trajectory_feature.map(trajectory_grading)
        # TODO: results under same target are partitioned, combine them.
        result_data = (stage_result
                       .join(latency_result)
                       .mapValues(merge_grading_results)
                       .join(zigzag_result)
                       .mapValues(merge_grading_results)
                       .join(trajectory_result)
                       .mapValues(merge_grading_results))
        logging.info(F'result_data_count: {result_data.count()}')
        logging.debug(F'result_data_first: {result_data.first()}')

        # 5. plot and visualize features, save grading result
        trajectory_feature.foreach(visual_utils.plot_hist)
        (result_data
         # PairRDD(target_prefix, combined_grading_result), output grading results for each target
         .foreach(output_result))
        self.email_output(todo_task_dirs.keys().collect(), origin_prefix, target_prefix)

    def email_output(self, tasks, origin_prefix, target_prefix, partner_email='', error_msg=''):
        title = 'Open Space Planner Profiling Results'
        # Uncomment this for dev test
        # recipients = email_utils.SIMULATION_TEAM
        recipients = email_utils.CONTROL_TEAM
        recipients.append(partner_email)
        SummaryTuple = namedtuple('Summary', ['Task', 'FeatureHDF5s', 'Profiling'])
        if tasks:
            email_content = []
            attachments = []
            tar_filename = None
            tar = None
            tasks.sort()
            for task in tasks:
                source = task.replace(target_prefix, origin_prefix, 1)
                logging.info(F'task: {task}, source: {source}')
                output = glob.glob(os.path.join(task, '*performance_grading*'))
                email_content.append(SummaryTuple(
                    Task=source,
                    FeatureHDF5s=len(glob.glob(os.path.join(task, '*.hdf5'))),
                    Profiling=len(output),
                ))
                if output:
                    tar_filename = F'{os.path.basename(source)}_profiling.tar.gz'
                    tar = tarfile.open(tar_filename, 'w:gz')
                    for report in output:
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
