#!/usr/bin/env python

"""Extracting features and grading the control performance based on the designed metrics"""

from collections import namedtuple
import glob
import os
import tarfile
import shutil

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
import fueling.profiling.feature_extraction.multi_job_control_feature_extraction_utils as feature_utils
import fueling.profiling.grading_evaluation.multi_job_control_performance_grading_utils as grading_utils

flags.DEFINE_string('ctl_metrics_input_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/control_profiling',
                    'input data directory for local run_test')
flags.DEFINE_string('ctl_metrics_output_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/control_profiling/generated',
                    'output data directory for local run_test')
flags.DEFINE_string('ctl_metrics_todo_tasks_local', '', 'todo_taks directory for local run_test')
flags.DEFINE_boolean('ctl_metrics_simulation_only_test', False,
                     'if simulation-only, then skip the owner/id/vehicle/controller identification')


class MultiJobControlProfilingMetrics(BasePipeline):
    """ Control Profiling: Feature Extraction and Performance Grading """

    def run_test(self):
        """Run test."""
        origin_prefix = flags.FLAGS.ctl_metrics_input_path_local
        if flags.FLAGS.ctl_metrics_simulation_only_test:
            target_prefix = flags.FLAGS.ctl_metrics_output_path_local
            todo_tasks = flags.FLAGS.ctl_metrics_todo_tasks_local.split(',')
            # RDD(tasks), the task dirs
            todo_task_dirs = self.to_rdd([
                os.path.join(origin_prefix, task) for task in todo_tasks
            ]).cache()
            logging.info(F'todo_task_dirs: {todo_task_dirs.collect()}')
        else:
            job_owner = self.FLAGS.get('job_owner')
            # Use year as the job_id, just for local test
            job_id = self.FLAGS.get('job_id')[:4]
            target_prefix = os.path.join(
                flags.FLAGS.ctl_metrics_output_path_local, job_owner, job_id)

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

            # # RDD(origin_vehicle_dir)
            todo_task_dirs = spark_helper.cache_and_log(
                'todo_jobs',
                origin_vehicle_dir
                # PairRDD(vehicle_type, list_of_records)
                .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*')))
                # RDD list_of_records to parse vehicle type and controller to
                # organize new key
                .values()
                .distinct())
            logging.info(F'todo_task_dirs: {todo_task_dirs.collect()}')

            conf_target_prefix = target_prefix
            logging.info(conf_target_prefix)
            generated_vehicle_dir = origin_vehicle_dir.mapValues(
                lambda path: path.replace(origin_prefix, conf_target_prefix, 1))
            logging.info(F'generated_vehicle_dir: {generated_vehicle_dir.collect()}')

            # PairRDD(source_vehicle_param_conf, dest_vehicle_param_conf))
            src_dst_rdd = origin_vehicle_dir.join(
                generated_vehicle_dir).values().cache()
            # Create dst dirs and copy conf file to them.
            src_dst_rdd.values().foreach(file_utils.makedirs)
            src_dst_rdd.foreach(
                lambda src_dst: shutil.copyfile(
                    os.path.join(
                        src_dst[0], feature_utils.CONF_FILE), os.path.join(
                        src_dst[1], feature_utils.CONF_FILE)))

        self.run(todo_task_dirs, origin_prefix, target_prefix)
        logging.info('Control Profiling Metrics: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = self.FLAGS.get(
            'input_data_path', 'modules/control/profiling/multi_job')

        job_owner = self.FLAGS.get('job_owner')
        # Use year as the job_id if data from apollo-platform, to avoid
        # processing same data repeatedly
        job_id = self.FLAGS.get('job_id') if self.is_partner_job() else self.FLAGS.get('job_id')[:4]
        job_email = partners.get(job_owner).email if self.is_partner_job() else ''
        logging.info(F'email address of job owner: {job_email}')

        target_prefix = os.path.join(dir_utils.inter_result_folder, job_owner, job_id)

        our_storage = self.our_storage()
        target_dir = our_storage.abs_path(target_prefix)
        # target_dir /mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-39-38
        logging.info(F'target_dir: {target_dir}')

        # Access partner's storage if provided.
        object_storage = self.partner_storage() or our_storage
        origin_dir = object_storage.abs_path(original_prefix)

        # origin_dir: our: /mnt/bos/modules/control/profiling/multi_job
        # partner: /mnt/partner/profiling/multi_job
        logging.info(F'origin_dir: {origin_dir}')

        # Sanity Check
        sanity_status = sanity_check(origin_dir,
                                     feature_utils.CONF_FILE, feature_utils.CHANNELS)
        if sanity_status is 'OK':
            logging.info('Sanity_Check: Passed.')
        else:
            logging.error(sanity_status)
            summarize_tasks([], origin_dir, target_dir, job_email, sanity_status)
            logging.info('Control Profiling Metrics: No Results, PROD')
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
        # [('Mkz7', 'modules/control/profiling/multi_job/Mkz7'), ...]
        # Partner [('Mkz7', 'profiling/multi_job/Mkz7'), ...]
        logging.info(F'origin_vehicle_dir: {origin_vehicle_dir.collect()}')

        # Copy vehicle parameter config file
        conf_target_prefix = target_dir
        target_vehicle_abs_dir = spark_helper.cache_and_log(
            'target_vehicle_abs_dir',
            origin_vehicle_dir
            .mapValues(object_storage.abs_path)
            .mapValues(lambda path: path.replace(origin_dir, conf_target_prefix, 1))
        )
        # target_vehicle_abs_dir:
        # [('Mkz7', '/mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-47-19/Mkz7'),...]
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
                os.path.join(
                    src_dst[0], feature_utils.CONF_FILE), os.path.join(
                    src_dst[1], feature_utils.CONF_FILE)))

        """ get to do jobs """
        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            # PairRDD(vehicle_type, relative_path_to_vehicle_type)
            origin_vehicle_dir
            # PairRDD(vehicle_type, files)
            .flatMapValues(object_storage.list_files)
            # PairRDD(vehicle_type, filterd absolute_path_to_records)
            .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or
                                          record_utils.is_bag_file(file)))
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
            # /Mkz7/Lon_Lat_Controller/Road_Test-2019-05-01/20190501110414'),...]
            logging.info(F'processed_dirs: {processed_dirs.collect()}')

            if not processed_dirs.isEmpty():
                """Reorgnize RDD key from vehicle/controller/record_prefix to vehicle=>abs target"""
                def _reorg_rdd_by_vehicle(target_task):
                    # parameter vehicle_controller_parsed like
                    # Mkz7/Lon_Lat_Controller/Road_Test-2019-05-01/20190501110414
                    vehicle, (vehicle_controller_parsed, task) = target_task
                    # vehicle = vehicle_controller_parsed.split('/')[0]
                    target_ = os.path.join(
                        target_dir, vehicle_controller_parsed)
                    return vehicle, target_

                target_vehicle_dir = spark_helper.cache_and_log(
                    'target_vehicle_dir',
                    origin_vehicle_dir
                    # PairRDD(vehicle_type, records)
                    .flatMapValues(object_storage.list_files)
                    # PairRDD(vehicle_type, filterd absolute_path_to_records)
                    .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or
                                                  record_utils.is_bag_file(file)))
                    # PairRDD(vehicle_type, absolute_path_to_records)
                    .mapValues(os.path.dirname)
                    # PairRDD(vehicle_controller_parsed, task_dir_with_target_prefix)
                    .mapValues(feature_utils.parse_vehicle_controller)
                    # PairRDD(vehicle_type, task_dir)
                    .map(_reorg_rdd_by_vehicle)
                    .distinct()
                )
                logging.info(F'target_vehicle_dir: {target_vehicle_dir.collect()}')

                todo_task_dirs = target_vehicle_dir.subtract(processed_dirs)

                logging.info(F'todo_tasks after subtracting: {todo_task_dirs.collect()}')
                # REMOVE CONTROLLER AND REPLACE ORIGIN PREFIX
                todo_task_dirs = spark_helper.cache_and_log(
                    'todo_task_dirs',
                    todo_task_dirs
                    # PairRDD(vehicle_type, directory replaced by origin_dir)
                    .mapValues(lambda dir: dir.replace(target_dir, origin_dir, 1))
                    # PairRDD(vehicle_type, origin directory)
                    .mapValues(multi_vehicle_utils.get_target_removed_controller)
                )
                logging.info(F'todo_tasks after postprocess: {todo_task_dirs.collect()}')

        logging.info(F'todo_tasks to run: {todo_task_dirs.values().collect()}')

        if not todo_task_dirs.collect():
            error_msg = 'No grading results: no new qualified data uploaded.'
            summarize_tasks([], origin_dir, target_dir, job_email, error_msg)
            logging.info('Control Profiling Metrics: No Results, PROD')
            return

        self.run(todo_task_dirs.values(), origin_dir, target_dir, job_email)
        logging.info('Control Profiling Metrics: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix, job_email=''):
        """Run the pipeline with given parameters"""

        """Reorgnize RDD key from vehicle/controller/record_prefix to absolute path"""
        def _reorg_target_dir(target_task):
            # parameter vehicle_controller_parsed like
            # Mkz7/Lon_Lat_Controller/Road_Test-2019-05-01/20190501110414
            vehicle_controller_parsed, task = target_task
            target_dir = os.path.join(target_prefix, vehicle_controller_parsed)
            return target_dir, task

        # RDD tasks
        reorganized_target = (todo_tasks
                              # PairRDD(vehicle_controller_parsed, tasks)
                              .map(feature_utils.parse_vehicle_controller)
                              # PairRDD(vehicle_controller_parsed, tasks)
                              .filter(spark_op.filter_value(lambda task: os.path.exists(task)))
                              # PairRDD(target_dir, task)
                              .map(_reorg_target_dir))

        logging.info(F'reorganized_target after reorg_target_dir:'
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
         .map(grading_utils.compute_h5_and_gradings)
         # PairRDD(target, combined_grading_result), combine gradings for each
         # target/task
         .reduceByKey(grading_utils.combine_gradings)
         # PairRDD(target, combined_grading_result), output grading results for
         # each target
         .foreach(grading_utils.output_gradings))

        reorganized_target_keys = reorganized_target.keys().collect()
        logging.info(F'reorganized_target: {reorganized_target_keys}')
        # Summarize by scanning the target directory
        summarize_tasks(reorganized_target_keys, original_prefix, target_prefix, job_email)

    def partition_data(self, target_msgs):
        """Divide the messages to groups each of which has exact number of messages"""
        logging.info(F'target messages: {target_msgs}')
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
    receivers = email_utils.DATA_TEAM + email_utils.CONTROL_TEAM
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
