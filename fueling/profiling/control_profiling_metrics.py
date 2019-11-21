#!/usr/bin/env python

"""Extracting features and grading the control performance based on the designed metrics"""

from collections import namedtuple
import glob
import os
import tarfile

from absl import flags
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.profiling.common.dir_utils as dir_utils
import fueling.profiling.feature_extraction.control_feature_extraction_utils as feature_utils
import fueling.profiling.grading_evaluation.control_performance_grading_utils as grading_utils


flags.DEFINE_string('ctl_metrics_input_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/control_profiling',
                    'input data directory for local run_test')
flags.DEFINE_string('ctl_metrics_output_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/control_profiling/generated',
                    'output data directory for local run_test')
flags.DEFINE_string('ctl_metrics_todo_tasks_local', '', 'todo_taks directory for local run_test')
flags.DEFINE_string('ctl_metrics_input_path_k8s', 'small-records/2019',
                    'input data directory for run_prod')
flags.DEFINE_string('ctl_metrics_output_path_k8s', 'modules/control/control_profiling_hf5',
                    'output data directory for run_pod')


class ControlProfilingMetrics(BasePipeline):
    """ Control Profiling: Feature Extraction and Performance Grading """

    def run_test(self):
        """Run test."""
        origin_prefix = flags.FLAGS.ctl_metrics_input_path_local
        target_prefix = flags.FLAGS.ctl_metrics_output_path_local
        # RDD(tasks), the task dirs
        todo_tasks = self.to_rdd([
            os.path.join(origin_prefix, flags.FLAGS.ctl_metrics_todo_tasks_local),
        ]).cache()
        self.run(todo_tasks, origin_prefix, target_prefix)
        summarize_tasks(todo_tasks.collect(), origin_prefix, target_prefix)
        logging.info('Control Profiling: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = flags.FLAGS.ctl_metrics_input_path_k8s
        target_prefix = flags.FLAGS.ctl_metrics_output_path_k8s
        # RDD(tasks), the task dirs
        todo_tasks = spark_helper.cache_and_log('todo_tasks',
                                                dir_utils.get_todo_tasks(original_prefix,
                                                                         target_prefix))
        self.run(todo_tasks, original_prefix, target_prefix)
        summarize_tasks(todo_tasks.collect(), original_prefix, target_prefix)
        logging.info('Control Profiling: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
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
                                                  record_utils.CHASSIS_CHANNEL,
                                                  record_utils.LOCALIZATION_CHANNEL]))
         # PairRDD(target_dir, (message)s)
         .groupByKey()
         # RDD(target_dir, group_id, group of (message)s), divide messages into groups
         .flatMap(partition_data)
         # PairRDD(target_dir, grading_result), for each group get the gradings and write h5 files
         .map(grading_utils.compute_h5_and_gradings)
         # PairRDD(target_dir, combined_grading_result), combine gradings for each target/task
         .reduceByKey(grading_utils.combine_gradings)
         # PairRDD(target_dir, combined_grading_result), output grading results for each target
         .foreach(grading_utils.output_gradings))


def partition_data(target_msgs):
    """Divide the messages to groups each of which has exact number of messages"""
    target, msgs = target_msgs
    logging.info('partition data for {} messages in target {}'.format(len(msgs), target))
    msgs = sorted(msgs, key=lambda msg: msg.timestamp)
    msgs_groups = [msgs[idx: idx + feature_utils.MSG_PER_SEGMENT]
                   for idx in range(0, len(msgs), feature_utils.MSG_PER_SEGMENT)]
    return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]


def summarize_tasks(tasks, original_prefix, target_prefix):
    """Make summaries to specified tasks"""
    SummaryTuple = namedtuple('Summary', ['Task', 'Records', 'HDF5s', 'Profling',
                                          'Primary_Gradings', 'Sample_Sizes'])
    title = 'Control Profiling Gradings Results'
    receivers = email_utils.DATA_TEAM + email_utils.CONTROL_TEAM
    email_content = []
    attachments = []
    target_dir_daily = None
    output_filename = None
    tar = None
    for task in tasks:
        target_dir = task.replace(original_prefix, target_prefix, 1)
        target_file = glob.glob(os.path.join(target_dir, '*performance_grading*'))
        scores, samples = grading_utils.highlight_gradings(task, target_file)
        email_content.append(SummaryTuple(
            Task=task,
            Records=len(glob.glob(os.path.join(task, '*record*'))),
            HDF5s=len(glob.glob(os.path.join(target_dir, '*.hdf5'))),
            Profling=len(glob.glob(os.path.join(target_dir, '*performance_grading*'))),
            Primary_Gradings=scores,
            Sample_Sizes=samples))
        if target_file:
            if target_dir_daily != os.path.dirname(target_dir):
                if output_filename and tar:
                    tar.close()
                    attachments.append(output_filename)
                target_dir_daily = os.path.dirname(target_dir)
                output_filename = os.path.join(target_dir_daily,
                                               '{}_gradings.tar.gz'
                                               .format(os.path.basename(target_dir_daily)))
                tar = tarfile.open(output_filename, 'w:gz')
            task_name = os.path.basename(target_dir)
            file_name = os.path.basename(target_file[0])
            tar.add(target_file[0], arcname='{}_{}'.format(task_name, file_name))
        file_utils.touch(os.path.join(target_dir, 'COMPLETE'))
    if tar:
        tar.close()
    attachments.append(output_filename)
    email_utils.send_email_info(title, email_content, receivers, attachments)


if __name__ == '__main__':
    ControlProfilingMetrics().main()
