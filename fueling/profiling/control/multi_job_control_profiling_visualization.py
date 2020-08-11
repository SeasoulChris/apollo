#!/usr/bin/env python

"""Visualize control features based on the designed metrics"""

from collections import namedtuple
import glob
import os
import tarfile
import time

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.common.partners import partners
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.spark_helper as spark_helper
import fueling.common.spark_op as spark_op
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils
import fueling.profiling.control.feature_visualization.control_feature_visualization_utils \
    as visual_utils


flags.DEFINE_boolean('ctl_visual_simulation_only_test', False,
                     'if simulation-only, then generate .json data file; otherwise, plotting')
flags.DEFINE_string('ctl_visual_simulation_vehicle', 'Mkz7',
                    'if simulation-only, then manually define the vehicle type in simulation')


class MultiJobControlProfilingVisualization(BasePipeline):
    """ Control Profiling: Visualize Control Features"""

    def run(self):
        """Visualization for both external and internal applications"""
        tic_start = time.perf_counter()

        if flags.FLAGS.ctl_visual_simulation_only_test:
            """Control Profiling: works on the 'auto-tuner + simulation' mode"""

            """Step 1: Path initialization to generate the input/output paths:"""
            #   (1) target_dir: abs path where the base_dir is task_name
            object_storage = self.our_storage()
            target_dir = object_storage.abs_path(flags.FLAGS.output_data_path)
            vehicle_type = flags.FLAGS.ctl_visual_simulation_vehicle

            """Step 2: Traverse files under input paths and generate todo_task paths:"""
            #   todo_tasks: key: vehicle_type
            #               value: abs path where the base_dir is task timestamp
            # PairRDD(target files)
            target_files = spark_helper.cache_and_log(
                'target_files',
                self.to_rdd([target_dir])
                # PairRDD(vehicle_type, [vehicle_type])
                .keyBy(lambda path: vehicle_type)
                # PairRDD(vehicle_type, file_path_under_vehicle_type)
                .flatMapValues(object_storage.list_files)
            )
            logging.info(F'target_files: {target_files.collect()}')

            # PairRDD(todo profiled dirs)
            todo_tasks = spark_helper.cache_and_log(
                'todo_tasks',
                target_files
                # PairRDD(vehicle_type, path)
                .mapValues(os.path.dirname)
                .distinct()
            )
            # if dirs have been graded before, then list them as follows
            logging.info(F'todo_tasks to run: {todo_tasks.collect()}')

            if not todo_tasks.collect():
                logging.info('Control Profiling Visualization: No Results')
                return

            """Step 3: Process data with profiling algorithm"""
            self.process(todo_tasks.values())

        else:
            """Control Visualization: works on the 'external/internal-user road-test' mode"""

            """Step 1: Path initialization to generate the input/output paths:"""
            #   (1) origin_dir: usually the same as target_dir
            #   (2) target_dir: abs path where the base_dir is data-processing date
            job_owner = self.FLAGS.get('job_owner')
            # Use year as the job_id if data from apollo-platform, to avoid
            # processing same data repeatedly
            job_id = (self.FLAGS.get('job_id') if self.is_partner_job() else
                      self.FLAGS.get('job_id')[:4])
            job_email = partners.get(job_owner).email if self.is_partner_job() else ''
            logging.info(F'email address of job owner: {job_email}')

            # obtain the target prefix
            output_prefix = (self.FLAGS.get('output_data_path')
                             or 'modules/control/tmp/results')
            target_prefix = os.path.join(output_prefix, job_owner, job_id)

            # For visualization, the object_storage always points to apollo storage.
            # by default, the target_dir is set to be the same as original_prefix
            object_storage = self.our_storage()
            target_dir = object_storage.abs_path(target_prefix)
            origin_dir = target_dir
            logging.info(F'target_dir: {target_dir}')

            if not os.path.isdir(origin_dir):
                error_msg = 'No visualization results: the source data path does not exist.'
                summarize_tasks([], origin_dir, target_dir, job_owner, job_email, error_msg)
                logging.info('Control Profiling Visualization: No Results, PROD')
                return

            """Step 2: Traverse files under input paths and generate todo_task paths:"""
            #   todo_tasks: key: vehicle_type
            #               value: abs path where the base_dir is task timestamp
            # PairRDD(target files)
            target_files = spark_helper.cache_and_log(
                'target_files',
                self.to_rdd([target_dir])
                # RDD([vehicle_type])
                .flatMap(multi_vehicle_utils.get_vehicle)
                # PairRDD(vehicle_type, [vehicle_type])
                .keyBy(lambda vehicle_type: vehicle_type)
                # PairRDD(vehicle_type, data_path_to_vehicle_type)
                .mapValues(lambda vehicle_type: os.path.join(target_prefix, vehicle_type))
                # PairRDD(vehicle_type, file_path_under_vehicle_type)
                .flatMapValues(object_storage.list_files)
            )
            logging.info(F'target_files: {target_files.collect()}')

            # PairRDD(processed plot dirs)
            processed_dirs = spark_helper.cache_and_log(
                'processed_dirs',
                target_files
                # PairRDD(vehicle_type, file endwith COMPLETE_PLOT)
                .filter(lambda key_path: key_path[1].endswith('COMPLETE_PLOT'))
                # PairRDD(vehicle_type, path)
                .mapValues(os.path.dirname)
                .distinct()
            )
            # if dirs have been visualized before, then list them as follows
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
            # if dirs have been graded before, then list them as follows
            logging.info(F'todo_tasks before filtering: {todo_tasks.collect()}')

            if not todo_tasks.collect():
                error_msg = 'No visualization results: no new qualified data uploaded.'
                summarize_tasks([], origin_dir, target_dir, job_owner, job_email, error_msg)
                logging.info('Control Profiling Visualization: No Results, PROD')
                return

            # todo_tasks = graded tasks - visualized tasks
            if not processed_dirs.isEmpty():
                todo_tasks = todo_tasks.subtract(processed_dirs)
            # if dirs have been graded but not visualized, then list them as:
            # /mnt/bos/modules/control/tmp/results/apollo/2019-11-25-10-47-19
            # /Mkz7/Lon_Lat_Controller/Road_Test-2019-05-01/20190501110414'
            logging.info(F'todo_tasks to run: {todo_tasks.collect()}')

            if not todo_tasks.collect():
                error_msg = 'No visualization results: all the data have been processed before.'
                summarize_tasks([], origin_dir, target_dir, job_owner, job_email, error_msg)
                logging.info('Control Profiling Visualization: No Results, PROD')
                return

            """Step 3: Process data with profiling algorithm"""
            self.process(todo_tasks.values())

            """Step 4: Summarize by scanning the target directory and send out emails"""
            summarize_tasks(todo_tasks.values().collect(), origin_dir, target_dir, job_owner, job_email)

        logging.info(f"Timer: total run() - {time.perf_counter() - tic_start: 0.04f} sec")
        logging.info('Control Profiling Visualization: All Done, PROD')

    def process(self, todo_tasks):
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

        if flags.FLAGS.ctl_visual_simulation_only_test:
            # PairRDD(target_dir, data_array)
            data_rdd.foreach(visual_utils.write_data_json_file)
        else:
            # PairRDD(target_dir, data_array)
            data_rdd.foreach(visual_utils.plot_h5_features_hist)


def summarize_tasks(tasks, original_prefix, target_prefix, job_owner, job_email='', error_msg=''):
    """Make summaries to specified tasks"""
    SummaryTuple = namedtuple(
        'Summary', ['Task', 'Target', 'HDF5s', 'VisualPlot'])
    title = 'Control Profiling Visualization Results for {}'.format(job_owner)
    receivers = email_utils.DATA_TEAM + email_utils.CONTROL_TEAM + email_utils.D_KIT_TEAM
    receivers.append(job_email)
    if tasks:
        email_content = []
        attachments = []
        target_dir_daily = None
        output_filename = None
        tar = None
        tasks.sort()
        for task in tasks:
            logging.info(F'task in summarize_tasks: {task}')
            target_dir = task
            target_file = glob.glob(os.path.join(target_dir, '*visualization*'))
            vehicle = task.replace(original_prefix, '', 1).split('/')[1]
            email_content.append(SummaryTuple(
                Task=task.replace(original_prefix, '', 1),
                Target=target_dir.replace(target_prefix, '', 1),
                HDF5s=len(glob.glob(os.path.join(task, '*.hdf5'))),
                VisualPlot=len(glob.glob(os.path.join(target_dir, '*visualization*')))))
            if target_file:
                if target_dir_daily != os.path.dirname(target_dir):
                    if output_filename and tar:
                        tar.close()
                        attachments.append(output_filename)
                    target_dir_daily = os.path.dirname(target_dir)
                    output_filename = os.path.join(target_dir_daily,
                                                   F'{vehicle}_'
                                                   F'{os.path.basename(target_dir_daily)}_'
                                                   F'plots.tar.gz')
                    tar = tarfile.open(output_filename, 'w:gz')
                task_name = os.path.basename(target_dir)
                file_name = os.path.basename(target_file[0])
                tar.add(target_file[0], arcname='{}_{}'.format(
                    task_name, file_name))
            file_utils.touch(os.path.join(target_dir, 'COMPLETE_PLOT'))
        if tar:
            tar.close()
        attachments.append(output_filename)
    else:
        logging.info('task in summarize_tasks: None')
        if error_msg:
            email_content = error_msg
        else:
            email_content = 'No visualization results: unknown reason.'
        attachments = []
    email_utils.send_email_info(title, email_content, receivers, attachments)


if __name__ == '__main__':
    MultiJobControlProfilingVisualization().main()
