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
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.profiling.common.dir_utils as dir_utils
import fueling.profiling.feature_visualization.control_feature_visualization_utils as visual_utils


flags.DEFINE_string('input_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/control_profiling/generated',
                    'input data directory for local run_test')
flags.DEFINE_string('output_path_local',
                    '/apollo/modules/data/fuel/testdata/profiling/control_profiling/generated',
                    'output data directory for local run_test')
flags.DEFINE_string('todo_tasks_local', '', 'todo_taks directory for local run_test')
flags.DEFINE_boolean('simulation_only_test', False,
                     'if simulation-only, then generate .json data file; otherwise, plotting')
flags.DEFINE_string('input_path_k8s', 'modules/control/control_profiling_hf5',
                    'input data directory for run_prod')
flags.DEFINE_string('output_path_k8s', 'modules/control/control_profiling_hf5',
                    'output data directory for run_pod')

class ControlProfilingVisualization(BasePipeline):
    """ Control Profiling: Visualize Control Features"""

    def run_test(self):
        """Run test."""
        origin_prefix = flags.FLAGS.input_path_local
        target_prefix = flags.FLAGS.output_path_local
        # RDD(tasks), the task dirs
        todo_tasks = self.to_rdd([
            os.path.join(origin_prefix, flags.FLAGS.todo_tasks_local),
        ]).cache()
        self.run(todo_tasks, origin_prefix, target_prefix)
        summarize_tasks(todo_tasks.collect(), origin_prefix, target_prefix)
        logging.info('Control Profiling Visualization: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = flags.FLAGS.input_path_k8s
        target_prefix = flags.FLAGS.output_path_k8s
        # RDD(tasks), the task dirs
        todo_tasks = spark_helper.cache_and_log('todo_tasks',
                                                dir_utils.get_todo_tasks(original_prefix, target_prefix,
                                                                         'COMPLETE', 'COMPLETE_PLOT'))
        self.run(todo_tasks, original_prefix, target_prefix)
        summarize_tasks(todo_tasks.collect(), original_prefix, target_prefix)
        logging.info('Control Profiling Visualization: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
        # RDD(tasks), with absolute paths
        data_rdd = (todo_tasks
                    # PairRDD(target_dir, task), the map of target dirs and source dirs
                    .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1))
                    # PairRDD(target_dir, task), filter out non-existed target dirs
                    .filter(spark_op.filter_key(os.path.isdir))
                    # PairRDD(target_dir, hdf5_file)
                    .mapValues(lambda task: glob.glob(os.path.join(task, '*.hdf5')))
                    # PairRDD(target_dir, list of data_array),
                    .mapValues(visual_utils.generate_segments)
                    # PairRDD(target_dir, data_array), by merging the arraies within the
                    # "segments" into one array
                    .mapValues(visual_utils.generate_data))
        # PairRDD(target_dir, data_array)
        data_rdd.foreach(visual_utils.write_data_json_file)
        # PairRDD(target_dir, data_array)
        data_rdd.foreach(visual_utils.plot_h5_features_hist)


def summarize_tasks(tasks, original_prefix, target_prefix):
    """Make summaries to specified tasks"""
    SummaryTuple = namedtuple('Summary', ['Task', 'Target', 'HDF5s', 'VisualPlot'])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    title = 'Control Profiling Visualization Results' + ' _ %s' % timestr
    receivers = email_utils.DATA_TEAM + email_utils.CONTROL_TEAM
    email_content = []
    attachments = []
    target_dir_daily = None
    output_filename = None
    tar = None
    for task in tasks:
        target_dir = task.replace(original_prefix, target_prefix, 1)
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
            tar.add(target_file[0], arcname='{}_{}'.format(task_name, file_name))
        file_utils.touch(os.path.join(target_dir, 'COMPLETE_PLOT'))
    if tar:
        tar.close()
    attachments.append(output_filename)
    email_utils.send_email_info(title, email_content, receivers, attachments)


if __name__ == '__main__':
    ControlProfilingVisualization().main()
