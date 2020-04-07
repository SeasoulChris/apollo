#!/usr/bin/env python

"""Extracting features and grading the control performance based on the designed metrics"""

from collections import namedtuple
import glob
import os
import shutil
import tarfile

from absl import flags
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
from fueling.common.partners import partners
import fueling.common.record_utils as record_utils
import fueling.profiling.common.dir_utils as dir_utils
from fueling.profiling.common.sanity_check import sanity_check
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.profiling.control.feature_extraction.multi_job_control_feature_extraction_utils \
    as feature_utils
import fueling.profiling.control.grading_evaluation.multi_job_control_performance_grading_utils \
    as grading_utils

flags.DEFINE_string('ctl_metrics_input_path', '', 'input data directory')
flags.DEFINE_string('ctl_metrics_output_path', 'modules/control/tmp/results',
                    'output data directory')
flags.DEFINE_string('ctl_metrics_todo_tasks', '', 'todo_tasks path')
flags.DEFINE_string('ctl_metrics_conf_path', '/mnt/bos/modules/control/control_conf',
                    'control conf storage path')

flags.DEFINE_boolean('ctl_metrics_simulation_only_test', False,
                     'if simulation-only, then execute the "Auto-Tuner + simulation" mode')
flags.DEFINE_string('ctl_metrics_simulation_vehicle', 'Mkz7',
                    'if simulation-only, then manually define the vehicle type in simulation')

flags.DEFINE_boolean('ctl_metrics_filter_by_MRAC', False,
                     'decide whether filtering out all the data without enabling MRAC control')
flags.DEFINE_string('ctl_metrics_weighted_score', 'MRAC_SCORE',
                    'select the score weighting method from control_channel_conf.py')


class MultiJobControlProfilingMetrics(BasePipeline):
    """ Control Profiling: Feature Extraction and Performance Grading """

    def run(self):
        """Work on actual road test data. Expect a single input directory"""

        if flags.FLAGS.ctl_metrics_simulation_only_test:
            """Control Profiling: works on the 'auto-tuner + simulation' mode"""
            origin_dir = flags.FLAGS.ctl_metrics_input_path
            target_dir = flags.FLAGS.ctl_metrics_output_path
            todo_tasks = flags.FLAGS.ctl_metrics_todo_tasks.split(',')
            vehicle_type = flags.FLAGS.ctl_metrics_simulation_vehicle
            job_email = ''
            # RDD(vehicle_type, tasks), the task dirs
            todo_task_dirs = self.to_rdd([
                os.path.join(origin_dir, task) for task in todo_tasks
            ]).keyBy(lambda dirs: vehicle_type).cache()
            logging.info(F'todo_task_dirs: {todo_task_dirs.collect()}')

            # RDD(target_vehicle_path), the target_vehicle dirs for .conf file
            generated_vehicle_dir = self.to_rdd([
                os.path.join(target_dir, vehicle_type)
            ]).cache()
            logging.info(F'generated_vehicle_dir: {generated_vehicle_dir.collect()}')

            # PairRDD(source_vehicle_param_conf, dest_vehicle_param_conf))
            src_dst_rdd = generated_vehicle_dir.keyBy(
                lambda path: os.path.join(self.FLAGS.get('ctl_metrics_conf_path'),
                                          os.path.basename(path).lower().replace(' ', '_', 1)))
            # Create dst dirs and copy conf file to them.
            src_dst_rdd.values().foreach(file_utils.makedirs)
            src_dst_rdd.foreach(
                lambda src_dst: shutil.copyfile(
                    os.path.join(src_dst[0], feature_utils.CONF_FILE),
                    os.path.join(src_dst[1], feature_utils.CONF_FILE)))
        else:
            """Control Profiling: works on the 'external/internal-user road-test' mode"""
            original_prefix = self.FLAGS.get('ctl_metrics_input_path') or self.FLAGS.get(
                'input_data_path', 'modules/control/profiling/multi_job')

            job_owner = self.FLAGS.get('job_owner')
            # Use year as the job_id if data from apollo-platform, to avoid
            # processing same data repeatedly
            job_id = (self.FLAGS.get('job_id') if self.is_partner_job() else
                      self.FLAGS.get('job_id')[:4])
            job_email = partners.get(job_owner).email if self.is_partner_job() else ''
            logging.info(F'email address of job owner: {job_email}')

            target_prefix = os.path.join(flags.FLAGS.ctl_metrics_output_path, job_owner, job_id)

            our_storage = self.our_storage()
            target_dir = our_storage.abs_path(target_prefix)
            # target_dir /mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-39-38
            logging.info(F'target_dir: {target_dir}')

            # Access partner's storage if provided.
            object_storage = self.partner_storage() or our_storage
            origin_dir = object_storage.abs_path(original_prefix)

            # origin_dir: our: /mnt/bos/modules/control/profiling/multi_job
            #             partner: /mnt/partner/profiling/multi_job
            logging.info(F'origin_dir: {origin_dir}')

            # Sanity Check
            sanity_status = sanity_check(origin_dir,
                                         feature_utils.CONF_FILE, feature_utils.CHANNELS)
            if sanity_status is 'OK':
                logging.info('Sanity_Check: Passed.')
            else:
                logging.error(sanity_status)
                summarize_tasks([], origin_dir, target_dir, job_email, sanity_status)
                logging.info('Control Profiling Metrics: No Results')
                return

            # RDD(origin_dir)
            origin_vehicle_dir = spark_helper.cache_and_log(
                'origin_vehicle_dir',
                self.to_rdd([origin_dir])
                # RDD([vehicle_type])
                .flatMap(multi_vehicle_utils.get_vehicle)
                # PairRDD(vehicle_type, [vehicle_type])
                .keyBy(lambda vehicle_type: vehicle_type)
                # PairRDD(vehicle_type, path_to_vehicle_type)
                .mapValues(lambda vehicle_type: os.path.join(original_prefix, vehicle_type)))
            # Our [('Mkz7', 'modules/control/profiling/multi_job/Mkz7'), ...]
            # Partner [('Mkz7', 'profiling/multi_job/Mkz7'), ...]
            logging.info(F'origin_vehicle_dir: {origin_vehicle_dir.collect()}')

            # Copy vehicle parameter config file
            target_vehicle_abs_dir = spark_helper.cache_and_log(
                'target_vehicle_abs_dir',
                origin_vehicle_dir
                .mapValues(object_storage.abs_path)
                .mapValues(lambda path: path.replace(origin_dir, target_dir, 1))
            )
            # target_vehicle_abs_dir: [('Mkz7',
            # '/mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-47-19/Mkz7'),...]
            logging.info(F'target_vehicle_abs_dir: {target_vehicle_abs_dir.collect()}')

            origin_vehicle_abs_dir = origin_vehicle_dir.mapValues(
                object_storage.abs_path)
            # PairRDD(origin_vehicle_abs_dir, dest_vehicle_abs_dir)
            src_dst_rdd = origin_vehicle_abs_dir.join(
                target_vehicle_abs_dir).values().cache()
            # src_dst_rdd: [('/mnt/bos/modules/control/profiling/multi_job/Mkz7',
            #  '/mnt/bos/modules/control/tmp/results/apollo/2019/Mkz7'),...]
            logging.info(F'src_dst_rdd: {src_dst_rdd.collect()}')

            # Create dst dirs and copy conf file to them.
            src_dst_rdd.values().foreach(file_utils.makedirs)
            src_dst_rdd.foreach(
                lambda src_dst: shutil.copyfile(
                    os.path.join(src_dst[0], feature_utils.CONF_FILE),
                    os.path.join(src_dst[1], feature_utils.CONF_FILE)))

            """ get todo jobs """
            todo_task_dirs = spark_helper.cache_and_log(
                'todo_jobs',
                # PairRDD(vehicle_type, relative_path_to_vehicle_type)
                origin_vehicle_dir
                # PairRDD(vehicle_type, files)
                .flatMapValues(object_storage.list_files)
                # PairRDD(vehicle_type, filterd absolute_path_to_records)
                .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file)
                                              or record_utils.is_bag_file(file)))
                # PairRDD(vehicle_type, absolute_path_to_records)
                .mapValues(os.path.dirname)
                .distinct()
            )
            logging.info(F'todo_task_dirs: {todo_task_dirs.collect()}')

            if not self.is_partner_job():
                processed_dirs = spark_helper.cache_and_log(
                    'processed_dirs',
                    self.to_rdd([target_dir])
                    .filter(spark_op.filter_value(lambda task: os.path.exists(task)))
                    # RDD([vehicle_type])
                    .flatMap(multi_vehicle_utils.get_vehicle)
                    # PairRDD(vehicle_type, [vehicle_type])
                    .keyBy(lambda vehicle_type: vehicle_type)
                    # PairRDD(vehicle_type, path_to_vehicle_type)
                    .mapValues(lambda vehicle_type: os.path.join(target_prefix, vehicle_type))
                    # PairRDD(vehicle_type, records)
                    .flatMapValues(object_storage.list_files)
                    # PairRDD(vehicle_type, file endwith COMPLETE)
                    .filter(lambda key_path: key_path[1].endswith('COMPLETE'))
                    # PairRDD(vehicle_type, path)
                    .mapValues(os.path.dirname)
                    .distinct()
                )
                # if processed same key before, result just like
                # [('Mkz7', '/mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-47-19
                # /Mkz7/Lon_Lat_Controller/2019-05-01/20190501110414'),...]
                logging.info(F'processed_dirs: {processed_dirs.collect()}')

                if not processed_dirs.isEmpty():
                    def _reorg_rdd_by_vehicle(target_task):
                        """Reorgnize RDD key from vehicle/controller/record_prefix """
                        """to vehicle=>abs target"""
                        # parameter vehicle_controller_parsed like
                        # Mkz7/Lon_Lat_Controller/2019-05-01/20190501110414
                        vehicle, (vehicle_controller_parsed, task) = target_task
                        # vehicle = vehicle_controller_parsed.split('/')[0]
                        target_ = os.path.join(target_dir, vehicle_controller_parsed)
                        return vehicle, target_

                    target_vehicle_dir = spark_helper.cache_and_log(
                        'target_vehicle_dir',
                        origin_vehicle_dir
                        # PairRDD(vehicle_type, records)
                        .flatMapValues(object_storage.list_files)
                        # PairRDD(vehicle_type, filterd absolute_path_to_records)
                        .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file)
                                                      or record_utils.is_bag_file(file)))
                        # PairRDD(vehicle_type, absolute_path_to_records)
                        .mapValues(os.path.dirname)
                        # PairRDD(vehicle_controller_parsed, task_dir_with_target_prefix)
                        .mapValues(lambda task:
                                   feature_utils.parse_vehicle_controller(task, self.FLAGS))
                        # PairRDD(vehicle_type, task_dir)
                        .map(_reorg_rdd_by_vehicle)
                        .distinct()
                    )
                    logging.info(F'target_vehicle_dir: {target_vehicle_dir.collect()}')

                    todo_task_dirs = target_vehicle_dir.subtract(processed_dirs)
                    logging.info(F'todo_task_dirs after subtracting: {todo_task_dirs.collect()}')

                    # REMOVE CONTROLLER AND REPLACE ORIGIN PREFIX
                    todo_task_dirs = spark_helper.cache_and_log(
                        'todo_task_dirs',
                        todo_task_dirs
                        # PairRDD(vehicle_type, directory replaced by origin_dir)
                        .mapValues(lambda dir: dir.replace(target_dir, origin_dir, 1))
                        # PairRDD(vehicle_type, origin directory)
                        .mapValues(multi_vehicle_utils.get_target_removed_controller)
                    )
                    logging.info(F'todo_task_dirs after postprocess: {todo_task_dirs.collect()}')

            logging.info(F'todo_tasks to run: {todo_task_dirs.values().collect()}')

            if not todo_task_dirs.collect():
                error_msg = 'No grading results: no new qualified data uploaded.'
                summarize_tasks([], origin_dir, target_dir, job_email, error_msg)

        if not todo_task_dirs.collect():
            logging.info('Control Profiling Metrics: No Results')
            return

        self.process(todo_task_dirs.values(), origin_dir, target_dir, job_email)
        logging.info('Control Profiling Metrics: All Done')

    def process(self, todo_tasks, original_prefix, target_prefix, job_email=''):
        """Run the pipeline with given parameters"""

        def _reorg_target_dir(target_task):
            """Reorgnize RDD key from vehicle/controller/record_prefix to absolute path"""
            # parameter vehicle_controller_parsed like
            # Mkz7/Lon_Lat_Controller/2019-05-01/20190501110414
            vehicle_controller_parsed, task = target_task
            target_dir = os.path.join(target_prefix, vehicle_controller_parsed)
            return target_dir, task

        # RDD tasks
        reorganized_target = (todo_tasks
                              # PairRDD(vehicle_controller_parsed, tasks)
                              .map(lambda task:
                                   feature_utils.parse_vehicle_controller(task, self.FLAGS))
                              # PairRDD(vehicle_controller_parsed, tasks)
                              .filter(spark_op.filter_value(lambda task: os.path.exists(task)))
                              # PairRDD(target_dir, task)
                              .map(_reorg_target_dir))

        logging.info(F'reorganized_target after _reorg_target_dir:'
                     F'{reorganized_target.collect()}')

        (reorganized_target
         # PairRDD(target_dir, record_file)
         .flatMapValues(lambda task: glob.glob(os.path.join(task, '*record*')) +
                        glob.glob(os.path.join(task, '*bag*')))
         # PairRDD(target_dir, record_file), filter out unqualified files
         .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or
                                       record_utils.is_bag_file(file)))
         # PairRDD(target_dir, message)
         .flatMapValues(record_utils.read_record(feature_utils.CHANNELS))
         #  # PairRDD(target_dir, messages)
         .groupByKey()
         # RDD(target, group_id, group of (message)s), divide messages into
         # groups
         .flatMap(self.partition_data)
         # PairRDD(target, grading_result), for each group get the gradings and
         # write h5 files
         .map(lambda target_groups:
              grading_utils.compute_h5_and_gradings(target_groups, self.FLAGS))
         # PairRDD(target, combined_grading_result), combine gradings for each
         # target/task
         .reduceByKey(grading_utils.combine_gradings)
         # PairRDD(target, combined_grading_result), output grading results for
         # each target
         .foreach(lambda grading_results:
                  grading_utils.output_gradings(grading_results, self.FLAGS)))

        reorganized_target_keys = reorganized_target.keys().collect()
        logging.info(F'reorganized_target: {reorganized_target_keys}')
        # Summarize by scanning the target directory
        if not flags.FLAGS.ctl_metrics_simulation_only_test:
            summarize_tasks(reorganized_target_keys, original_prefix, target_prefix, job_email)

    def partition_data(self, target_msgs):
        """Divide the messages to groups each of which has exact number of messages"""
        target, msgs = target_msgs

        logging.info(
            F'partition data for {len(msgs)} messages in target: {target}')
        msgs = sorted(msgs, key=lambda msg: msg.timestamp)
        msgs_groups = [msgs[idx: idx + feature_utils.MSG_PER_SEGMENT]
                       for idx in range(0, len(msgs), feature_utils.MSG_PER_SEGMENT)]
        return [(target, group_id, group)
                for group_id, group in enumerate(msgs_groups)]


def summarize_tasks(targets, original_prefix, target_prefix, job_email='', error_msg=''):
    """Make summaries to specified tasks"""
    SummaryTuple = namedtuple(
        'Summary', [
            'Task', 'Records', 'HDF5s', 'Profling', 'Primary_Gradings', 'Sample_Sizes'])
    title = 'Control Profiling Gradings Results'
    receivers = email_utils.DATA_TEAM + email_utils.CONTROL_TEAM + email_utils.D_KIT_TEAM
    receivers.append(job_email)
    if targets:
        email_content = []
        attachments = []
        target_dir_daily = None
        output_filename = None
        tar = None
        targets.sort()
        for target_dir in targets:
            logging.info(F'target_dir in summarize_tasks: {target_dir}')
            target_postfix = target_dir.replace(target_prefix, '', 1)
            vehicle = target_postfix.split('/')[1]
            controller = target_postfix.split('/')[2]
            task = original_prefix + target_postfix.replace('/' + controller, '', 1)
            logging.info(F'task_dir in summarize_tasks: {task}')
            target_file = glob.glob(os.path.join(
                target_dir, '*performance_grading*'))
            target_conf = glob.glob(os.path.join(
                target_dir, '*control_profiling_conf*'))
            scores, samples = grading_utils.highlight_gradings(
                task, target_file)
            email_content.append(SummaryTuple(
                Task=task,
                Records=len(glob.glob(os.path.join(task, '*record*')))
                + len(glob.glob(os.path.join(task, '*bag*'))),
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
                    output_filename = os.path.join(
                        target_dir_daily,
                        F'{vehicle}_{os.path.basename(target_dir_daily)}_gradings.tar.gz')
                    tar = tarfile.open(output_filename, 'w:gz')
                task_name = os.path.basename(target_dir)
                file_name = os.path.basename(target_file[0])
                conf_name = os.path.basename(target_conf[0])
                tar.add(target_file[0], arcname=F'{task_name}_{file_name}')
                tar.add(target_conf[0], arcname=F'{task_name}_{conf_name}')
            file_utils.touch(os.path.join(target_dir, 'COMPLETE'))
        if tar:
            tar.close()
        attachments.append(output_filename)
    else:
        logging.info('target_dir in summarize_tasks: None')
        if error_msg:
            email_content = error_msg
        else:
            email_content = 'No grading results: unknown reason.'
        attachments = []
    email_utils.send_email_info(
        title, email_content, receivers, attachments)


if __name__ == '__main__':
    MultiJobControlProfilingMetrics().main()
