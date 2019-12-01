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
from fueling.common.storage.bos_client import BosClient
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
        target_prefix = '/apollo/modules/data/fuel/testdata/profiling/multi_job_genanrated/{}/{}'\
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

        logging.info('todo_task_dirs %s' % todo_task_dirs.collect())

        conf_target_prefix = target_prefix
        logging.info(conf_target_prefix)
        generated_vehicle_dir = origin_vehicle_dir.mapValues(
            lambda path: path.replace(origin_prefix, conf_target_prefix, 1))
        logging.info('generated_vehicle_dir: %s' % generated_vehicle_dir.collect())

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
        logging.info('Control Profiling: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = self.FLAGS.get(
            'input_data_path', 'modules/control/profiling/multi_job')

        job_owner = self.FLAGS.get('job_owner')
        # Use year as the job_id if data from apollo-platform, to avoid
        # processing same data repeatedly
        job_id = self.FLAGS.get('job_id') if self.is_partner_job() else self.FLAGS.get('job_id')[:4]
        target_prefix = os.path.join(dir_utils.inter_result_folder, job_owner, job_id)

        bucket_apollo_platform = BosClient()
        target_dir = bucket_apollo_platform.abs_path(target_prefix)
        # target_dir /mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-39-38
        logging.info('target_dir %s' % target_dir)

        # Access partner's storage if provided.
        object_storage = self.partner_storage() or bucket_apollo_platform
        origin_dir = object_storage.abs_path(original_prefix)

        # origin_dir: our: /mnt/bos/modules/control/profiling/multi_job
        # partner: /mnt/partner/profiling/multi_job
        logging.info("origin_dir: %s" % origin_dir)

        # Sanity Check
        if not sanity_check(
                origin_dir,
                feature_utils.CONF_FILE,
                feature_utils.CHANNELS,
                job_owner,
                job_id):
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
        logging.info('origin_vehicle_dir: %s' % origin_vehicle_dir.collect())

        # Copy vehicle parameter config file
        conf_target_prefix = target_dir
        logging.info(conf_target_prefix)
        target_vehicle_dir = origin_vehicle_dir
            .map(lambda path: object_storage.abs_path)
            .mapValues(lambda path: path.replace(origin_dir, conf_target_prefix, 1))
        # target_vehicle_dir:
        # [('Mkz7', '/mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-47-19/Mkz7'),...]
        logging.info('target_vehicle_dir: %s' % target_vehicle_dir.collect())

        # PairRDD(source_vehicle_param_conf, dest_vehicle_param_conf))
        src_dst_rdd = origin_vehicle_dir.join(target_vehicle_dir).values().cache()
        # touch target dir avoiding to copy failed
        file_utils.makedirs(target_dir)
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
        )

        todo_task_dirs = spark_helper.cache_and_log(
            'todo_jobs',
            todo_task_dirs
            # PairRDD(vehicle_type, filterd absolute_path_to_records)
            .filter(spark_op.filter_value(lambda file: record_utils.is_record_file(file) or
                                          record_utils.is_bag_file(file)))
            # PairRDD(vehicle_type, absolute_path_to_records)
            .mapValues(os.path.dirname)
            .distinct()
        )

        logging.info('todo_task_dirs %s' % todo_task_dirs.collect())

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
            logging.info('processed_dirs: %s' % processed_dirs.collect())

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
                logging.info('target_vehicle_dir %s' %
                             target_vehicle_dir.collect())

                todo_task_dirs = target_vehicle_dir.subtract(processed_dirs)

                logging.info('todo_tasks after substrct %s' %
                             todo_task_dirs.collect())
                # REMOVE CONTROLLER AND REPLACE ORIGIN PREFIX
                todo_task_dirs = spark_helper.cache_and_log(
                    'todo_task_dirs',
                    todo_task_dirs
                    # PairRDD(vehicle_type, directory replaced by origin_dir)
                    .mapValues(lambda dir: dir.replace(target_dir, origin_dir, 1))
                    # PairRDD(vehicle_type, origin directory)
                    .mapValues(multi_vehicle_utils.get_target_removed_controller)
                )
                logging.info('todo_tasks after postprocess %s' %
                             todo_task_dirs.collect())

        logging.info('todo_tasks to run %s' %
                     todo_task_dirs.values().collect())

        self.run(todo_task_dirs.values(), original_prefix, target_dir)
        logging.info('Control Profiling: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""

        """Reorgnize RDD key from vehicle/controller/record_prefix to absolute path"""
        def _reorg_target_dir(target_task):
            # parameter vehicle_controller_parsed like
            # Mkz7/Lon_Lat_Controller/Road_Test-2019-05-01/20190501110414
            vehicle_controller_parsed, task = target_task
            target_dir = os.path.join(target_prefix, vehicle_controller_parsed)
            return target_dir, task

        # RDD tasks
        reorgnized_target = (todo_tasks
                             # PairRDD(vehicle_controller_parsed, tasks)
                             .map(feature_utils.parse_vehicle_controller)
                             # PairRDD(vehicle_controller_parsed, tasks)
                             .filter(spark_op.filter_value(lambda task: os.path.exists(task)))
                             # PairRDD(target_dir, task)
                             .map(_reorg_target_dir))

        logging.info('reorgnized_target after reorg_target_dir:%s' %
                     reorgnized_target.collect())

        (reorgnized_target
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

        logging.info('reorgnized_target:%s' %
                     reorgnized_target.keys().collect())
        # Summarize by new tasks contains Controller type
        self.summarize_tasks(reorgnized_target.keys().collect(),
                             original_prefix, target_prefix)

    def partition_data(self, target_msgs):
        """Divide the messages to groups each of which has exact number of messages"""
        logging.info('target messages:{}'.format(target_msgs))
        target, msgs = target_msgs

        logging.info(
            'partition data for {} messages in target {}'.format(
                len(msgs), target))
        msgs = sorted(msgs, key=lambda msg: msg.timestamp)
        msgs_groups = [msgs[idx: idx + feature_utils.MSG_PER_SEGMENT]
                       for idx in range(0, len(msgs), feature_utils.MSG_PER_SEGMENT)]
        return [(target, group_id, group)
                for group_id, group in enumerate(msgs_groups)]

    def summarize_tasks(self, tasks, original_prefix, target_prefix):
        """Make summaries to specified tasks"""
        SummaryTuple = namedtuple(
            'Summary', [
                'Task', 'Records', 'HDF5s', 'Profling', 'Primary_Gradings', 'Sample_Sizes'])
        title = 'Control Profiling Gradings Results'
        receivers = email_utils.DATA_TEAM + email_utils.CONTROL_TEAM
        email_content = []
        attachments = []
        target_dir_daily = None
        output_filename = None
        tar = None
        for task in tasks:
            logging.info('processing task is {}'.format(task))
            target_dir = task
            logging.warning(
                'target_dir in summarize_tasks :{}'.format(target_dir))

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
                    output_filename = os.path.join(
                        target_dir_daily, '{}_gradings.tar.gz' .format(
                            os.path.basename(target_dir_daily)))
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
