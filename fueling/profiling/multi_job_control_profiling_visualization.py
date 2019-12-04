#!/usr/bin/env python

"""Visualize control features based on the designed metrics"""

from collections import namedtuple
import glob
import os
import sys
import tarfile
import time

from absl import flags
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.common.storage.bos_client import BosClient
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.profiling.common.dir_utils as dir_utils
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.profiling.feature_visualization.control_feature_visualization_utils as visual_utils


flags.DEFINE_string('ctl_visual_input_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/multi_job_genanrated',
                    'input data directory for local run_test')
flags.DEFINE_string('ctl_visual_output_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/multi_job_genanrated',
                    'output data directory for local run_test')
flags.DEFINE_string('ctl_visual_todo_tasks_local', '',
                    'todo_taks directory for local run_test')
flags.DEFINE_boolean('ctl_visual_simulation_only_test', False,
                     'if simulation-only, then generate .json data file; otherwise, plotting')
flags.DEFINE_string('ctl_visual_input_path_k8s', 'modules/control/tmp/results',
                    'input data directory for run_prod')
flags.DEFINE_string('ctl_visual_output_path_k8s', 'modules/control/tmp/results',
                    'output data directory for run_pod')


class MultiJobControlProfilingVisualization(BasePipeline):
    """ Control Profiling: Visualize Control Features"""

    def run_test(self):
        """Run test."""

        origin_prefix = flags.FLAGS.ctl_visual_input_path_local
        target_prefix = flags.FLAGS.ctl_visual_output_path_local

        if flags.FLAGS.ctl_visual_simulation_only_test:
            todo_tasks_postfix = flags.FLAGS.ctl_visual_todo_tasks_local.split(
                ',')
            # RDD(tasks), the task dirs
            todo_tasks = self.to_rdd([
                os.path.join(origin_prefix, task) for task in todo_tasks_postfix
            ]).cache()
        else:
            job_owner = self.FLAGS.get('job_owner')
            # Use year as the job_id, just for local test
            job_id = self.FLAGS.get('job_id')[:4]
            origin_prefix = os.path.join(
                flags.FLAGS.ctl_visual_input_path_local, job_owner, job_id)
            target_prefix = os.path.join(
                flags.FLAGS.ctl_visual_output_path_local, job_owner, job_id)
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
                .mapValues(lambda vehicle: os.path.join(origin_prefix, vehicle))
            )

        # RDD(origin_vehicle_dir)
            todo_tasks = spark_helper.cache_and_log(
                'todo_tasks',
                origin_vehicle_dir
                # PairRDD(vehicle_type, list_of_records)
                .flatMapValues(lambda path: glob.glob(os.path.join(path, '*/*/*')))
                # RDD list_of_records to parse vehicle type and controller to
                # organize new key
                .filter(spark_op.filter_value(lambda task: os.path.isdir(task)))
                .values()
                .distinct()
            )

        logging.info(F'todo_tasks {todo_tasks.collect()}')

        self.run(todo_tasks, origin_prefix, target_prefix)
        summarize_tasks(todo_tasks.collect(), origin_prefix, target_prefix)
        logging.info('Control Profiling Visualization: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        job_owner = self.FLAGS.get('job_owner')
        # Use year as the job_id if data from apollo-platform, to avoid
        # processing same data repeatedly
        job_id = self.FLAGS.get('job_id') if self.is_partner_job(
        ) else self.FLAGS.get('job_id')[:4]
        # same origin and target prefix
        original_prefix = os.path.join(
            flags.FLAGS.ctl_visual_input_path_k8s, job_owner, job_id)
        target_prefix = os.path.join(
            flags.FLAGS.ctl_visual_output_path_k8s, job_owner, job_id)

        # In visualization application, the object_storage always points to apollo storage.
        object_storage = self.our_storage()
        target_dir = object_storage.abs_path(target_prefix)
        logging.info(F'target_dir {target_dir}')

        origin_dir = object_storage.abs_path(original_prefix)
        logging.info(F'origin_dir {origin_dir}')

        # PairRDD(target files)
        target_files = spark_helper.cache_and_log(
            'target_files',
            self.to_rdd([target_dir])
            # .filter(spark_op.filter_value(os.path.isdir))
            # RDD([vehicle_type])
            .flatMap(multi_vehicle_utils.get_vehicle)
            # PairRDD(vehicle_type, [vehicle_type])
            .keyBy(lambda vehicle_type: vehicle_type)
            # PairRDD(vehicle_type, path_to_vehicle_type)
            .mapValues(lambda vehicle_type: os.path.join(target_prefix, vehicle_type))
            # PairRDD(vehicle_type, records)
            .flatMapValues(object_storage.list_files)
        )
        logging.info(F'target_files: {target_files.collect()}')

        # PairRDD(processed plot dirs)
        processed_dirs = spark_helper.cache_and_log(
            'processed_dirs',
            target_files
            # PairRDD(vehicle_type, file endwith COMPLETED)
            .filter(lambda key_path: key_path[1].endswith('COMPLETE_PLOT'))
            # PairRDD(vehicle_type, path)
            .mapValues(os.path.dirname)
            .distinct()
        )
        # if processed same key before, result just like
        # /mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-47-19
        # /Mkz7/Lon_Lat_Controller/Road_Test-2019-05-01/20190501110414'
        logging.info(F'processed_dirs: {processed_dirs.collect()}')

        # PairRDD(todo profiled dirs)
        todo_tasks = spark_helper.cache_and_log(
            'todo_tasks',
            target_files
            # PairRDD(vehicle_type, file endwith COMPLETE)
            .filter(lambda key_path: key_path[1].endswith('COMPLETE'))
            # PairRDD(vehicle_type, path)
            .mapValues(os.path.dirname)
            .distinct()
        )
        logging.info(F'todo_tasks before filtering: {todo_tasks.collect()}')

        if not processed_dirs.isEmpty():
            todo_tasks = todo_tasks.subtract(processed_dirs)

        logging.info(F'todo_tasks to run: {todo_tasks.values().collect()}')

        self.run(todo_tasks.values(), origin_dir, target_dir)
        summarize_tasks(todo_tasks.values().collect(), origin_dir, target_dir)
        logging.info('Control Profiling Visualization: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
        # RDD(tasks), with absolute paths
        data_rdd = (todo_tasks
                    # PairRDD(target_dir, task), the map of target dirs and source dirs
                    .keyBy(lambda source: source)
                    # PairRDD(target_dir, task), filter out non-existed target dirs
                    .filter(spark_op.filter_key(os.path.isdir))
                    # PairRDD(target_dir, hdf5_file)
                    .mapValues(lambda task: glob.glob(os.path.join(task, '*.hdf5')))
                    # PairRDD(target_dir, list of data_array),
                    .mapValues(visual_utils.generate_segments)
                    # PairRDD(target_dir, data_array), by merging the arraies within the
                    # "segments" into one array
                    .mapValues(visual_utils.generate_data)
                    )
        logging.info(F'data_rdd {data_rdd.collect()}')

        if flags.FLAGS.ctl_visual_simulation_only_test:
            # PairRDD(target_dir, data_array)
            data_rdd.foreach(visual_utils.write_data_json_file)
        else:
            # PairRDD(target_dir, data_array)
            data_rdd.foreach(visual_utils.plot_h5_features_hist)


def summarize_tasks(tasks, original_prefix, target_prefix):
    """Make summaries to specified tasks"""
    SummaryTuple = namedtuple(
        'Summary', ['Task', 'Target', 'HDF5s', 'VisualPlot'])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    title = 'Control Profiling Visualization Results' + ' _ %s' % timestr
    receivers = email_utils.DATA_TEAM + email_utils.CONTROL_TEAM
    email_content = []
    attachments = []
    target_dir_daily = None
    output_filename = None
    tar = None
    for task in tasks:
        logging.info(F'task in summarize_tasks {task}')
        target_dir = task
        target_file = glob.glob(os.path.join(target_dir, '*visualization*'))
        email_content.append(SummaryTuple(
            Task=task,
            Target=target_dir,
            HDF5s=len(glob.glob(os.path.join(task, '*.hdf5'))),
            VisualPlot=len(glob.glob(os.path.join(target_dir, '*visualization*')))))
        if target_file:
            if target_dir_daily != os.path.dirname(target_dir):
                if output_filename and tar:
                    tar.close()
                    attachments.append(output_filename)
                target_dir_daily = os.path.dirname(target_dir)
                output_filename = os.path.join(target_dir_daily,
                                               '{}_plots.tar.gz'
                                               .format(os.path.basename(target_dir_daily)))
                tar = tarfile.open(output_filename, 'w:gz')
            task_name = os.path.basename(target_dir)
            file_name = os.path.basename(target_file[0])
            tar.add(target_file[0], arcname='{}_{}'.format(
                task_name, file_name))
        file_utils.touch(os.path.join(target_dir, 'COMPLETE_PLOT'))
    if tar:
        tar.close()
    attachments.append(output_filename)
    email_utils.send_email_info(title, email_content, receivers, attachments)


if __name__ == '__main__':
    MultiJobControlProfilingVisualization().main()
