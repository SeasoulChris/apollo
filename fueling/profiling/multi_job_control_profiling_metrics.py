#!/usr/bin/env python

"""Extracting features and grading the control performance based on the designed metrics"""

from collections import namedtuple
import glob
import os
import tarfile
import shutil

import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.record_utils as record_utils
import fueling.profiling.common.dir_utils as dir_utils
from fueling.profiling.common.sanity_check import sanity_check
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.profiling.feature_extraction.multi_job_control_feature_extraction_utils as feature_utils
import fueling.profiling.grading_evaluation.multi_job_control_performance_grading_utils as grading_utils


class ControlProfilingMetrics(BasePipeline):
    """ Control Profiling: Feature Extraction and Performance Grading """

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/profiling/multi_job'
        target_prefix = '/apollo/modules/data/fuel/testdata/profiling/multi_job/genanrated/{}/{}'\
            .format(self.FLAGS.get('job_owner'),
                    self.FLAGS.get('job_id'))

        """origin vehicle directory"""
        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([origin_prefix])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, vehicle_type)
            .keyBy(lambda vehicle: vehicle)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle)))

        logging.info('vehicle directory:{}'.format(
            origin_vehicle_dir.collect()))

        # # RDD(origin_vehicle_dir)
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            origin_vehicle_dir
            # PairRDD(vehicle_type, list_of_records)
            .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*')))
            .values()
            .distinct())

        logging.info('todo_task_dirs %s' % todo_task_dirs.collect())

        conf_target_prefix = target_prefix
        logging.info(conf_target_prefix)
        target_param_conf = origin_vehicle_dir.mapValues(
            lambda path: path.replace(origin_prefix, conf_target_prefix, 1))
        logging.info('target_param_conf: %s' % target_param_conf.collect())

        # PairRDD(source_vehicle_param_conf, dest_vehicle_param_conf))
        src_dst_rdd = origin_vehicle_dir.join(
            target_param_conf).values().cache()
        # Create dst dirs and copy conf file to them.
        src_dst_rdd.values().foreach(file_utils.makedirs)
        src_dst_rdd.foreach(lambda src_dst: shutil.copyfile(os.path.join(src_dst[0],
                                                                         feature_utils.CONF_FILE),
                                                            os.path.join(src_dst[1],
                                                                         feature_utils.CONF_FILE)))

        self.run(todo_task_dirs, origin_prefix, target_prefix)
        self.summarize_tasks(todo_task_dirs.collect(),
                             origin_prefix, target_prefix)
        logging.info('Control Profiling: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        # test online /mnt/bos/modules/control/apollo_calibration_table/Mkz7/2019-05-31/
        original_prefix = 'small-records/2019'
        target_prefix = '{}/{}/modules/control/control_profiling_hf5'.format(
            self.FLAGS.get('job_owner'), self.FLAGS.get('job_id'))

        # RDD(origin_dir)
        origin_vehicle_dir = spark_helper.cache_and_log(
            'origin_vehicle_dir',
            self.to_rdd([original_prefix])
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(original_prefix, vehicle_type)))

        # RDD(tasks), the task dirs
        todo_tasks = spark_helper.cache_and_log(
            'todo_tasks',
            dir_utils.get_todo_tasks(original_prefix, target_prefix))

        if not sanity_check(origin_vehicle_dir, feature_utils.CONF_FILE, feature_utils.CHANNELS,
                            self.FLAGS.get['job_owner'], self.FLAGS.get['job_id']):
            return

        conf_target_prefix = target_prefix
        logging.info(conf_target_prefix)
        target_param_conf = origin_vehicle_dir.mapValues(
            lambda path: path.replace(original_prefix, conf_target_prefix, 1))
        logging.info('target_param_conf: %s' % target_param_conf.collect())
        print("origin_vehicle_dir.join", origin_vehicle_dir.join(
            target_param_conf).collect())

        # PairRDD(source_vehicle_param_conf, dest_vehicle_param_conf))
        src_dst_rdd = origin_vehicle_dir.join(
            target_param_conf).values().cache()
        # Create dst dirs and copy conf file to them.
        src_dst_rdd.values().foreach(file_utils.makedirs)
        src_dst_rdd.foreach(lambda src_dst: shutil.copyfile(os.path.join(src_dst[0], feature_utils.CONF_FILE),
                                                            os.path.join(src_dst[1], feature_utils.CONF_FILE)))

        self.run(todo_tasks, original_prefix, target_prefix)
        self.summarize_tasks(todo_tasks.collect(),
                             original_prefix, target_prefix)
        logging.info('Control Profiling: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""

        # PairRDD (vehicle , tasks)
        (todo_tasks
         # PairRDD(target, tasks)
         .map(feature_utils.verify_vehicle_controller)
         # PairRDD(target, record_file)
         .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')) +
                        glob.glob(os.path.join(task, '*bag*')))
         # PairRDD(target_dir, record_file), filter out unqualified files
         .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or
                                       record_utils.is_bag_file(file)))
         # PairRDD(target_dir, message)
         .flatMapValues(record_utils.read_record(feature_utils.CHANNELS))
         #  # PairRDD(target_dir, messages)
         .groupByKey()
         # RDD(target, group_id, group of (message)s), divide messages into groups
         .flatMap(self.partition_data)
         # PairRDD(target, grading_result), for each group get the gradings and write h5 files
         .map(grading_utils.compute_h5_and_gradings)
         # PairRDD(target, combined_grading_result), combine gradings for each target/task
         .reduceByKey(grading_utils.combine_gradings)
         # PairRDD(target, combined_grading_result), output grading results for each target
         .foreach(grading_utils.output_gradings))

    def partition_data(self, target_msgs):
        """Divide the messages to groups each of which has exact number of messages"""
        logging.info('target messages:{}'.format(target_msgs))
        target, msgs = target_msgs

        logging.info(
            'partition data for {} messages in target {}'.format(len(msgs), target))
        msgs = sorted(msgs, key=lambda msg: msg.timestamp)
        msgs_groups = [msgs[idx: idx + feature_utils.MSG_PER_SEGMENT]
                       for idx in range(0, len(msgs), feature_utils.MSG_PER_SEGMENT)]
        return [(target, group_id, group) for group_id, group in enumerate(msgs_groups)]

    def summarize_tasks(self, tasks, original_prefix, target_prefix):
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
            logging.info('processing task is {}'.format(task))
            # TODO: TARGET_DIR NEED UPDATE here
            task = task[1]
            target_dir = task.replace(original_prefix, target_prefix, 1)
            logging.warning('target_dir in summarize_tasks :{}'.format(target_dir))

            target_file = glob.glob(os.path.join(
                target_dir, '*performance_grading*'))
            scores, samples = grading_utils.highlight_gradings(
                task, target_file)
            email_content.append(SummaryTuple(
                Task=task,
                Records=len(glob.glob(os.path.join(task, '*record*'))),
                HDF5s=len(glob.glob(os.path.join(target_dir, '*.hdf5'))),
                Profling=len(glob.glob(os.path.join(
                    target_dir, '*performance_grading*'))),
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
                tar.add(target_file[0], arcname='{}_{}'.format(
                    task_name, file_name))
            file_utils.touch(os.path.join(target_dir, 'COMPLETE'))
        if tar:
            tar.close()
        attachments.append(output_filename)
        email_utils.send_email_info(
            title, email_content, receivers, attachments)


if __name__ == '__main__':
    ControlProfilingMetrics().main()
