#!/usr/bin/env python

"""Visualize control features based on the designed metrics"""

from collections import namedtuple
import glob
import os
import time

import colored_glog as glog
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.s3_utils as s3_utils
import fueling.control.control_profiling.offline_visualization.control_feature_visualization_utils \
       as visual_utils


class ControlProfilingVisualization(BasePipeline):
    """ Control Profiling: Visualize Control Features"""

    def __init__(self):
        """ initialize """
        BasePipeline.__init__(self, 'control_profiling_visualization')

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/control/control_profiling/generated'
        target_prefix = origin_prefix
        # RDD(tasks), the task dirs
        todo_tasks = self.context().parallelize([
            os.path.join(origin_prefix, 'Transit_Auto'),
            os.path.join(origin_prefix, 'Transit_Auto2')
        ]).cache()
        self.run(todo_tasks, origin_prefix, target_prefix)
        summarize_tasks(todo_tasks.collect(), origin_prefix, target_prefix)
        glog.info('Control Profiling Visualization: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = 'modules/control/control_profiling_hf5'
        target_prefix = original_prefix
        # RDD(tasks), the task dirs
        todo_tasks = spark_helper.cache_and_log('todo_tasks',
            get_todo_tasks(original_prefix, target_prefix))
        self.run(todo_tasks, original_prefix, target_prefix)
        summarize_tasks(todo_tasks.collect(), original_prefix, target_prefix)
        glog.info('Control Profiling Visualization: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
        # RDD(tasks), with absolute paths
        (todo_tasks
        # PairRDD(target_dir, task), the map of target dirs and source dirs
        .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1))
        # PairRDD(target_dir, task), filter out non-existed target dirs
        .filter(spark_op.filter_key(os.path.isdir))
        # PairRDD(target_dir, hdf5_file)
        .mapValues(lambda task: glob.glob(os.path.join(task, '*.hdf5')))
        # PairRDD(target_dir, list of data_array),
        .mapValues(visual_utils.generate_segments)
        # PairRDD(target_dir, data_array), by merging the arraies within the "segments" into one array
        .mapValues(visual_utils.generate_data)
        # PairRDD(target_dir, data_array)
        .foreach(visual_utils.plot_h5_features_hist))

def get_todo_tasks(origin_prefix, target_prefix,
                   marker_origin='COMPLETE', marker_processed='COMPLETE'):
    """Get to be processed tasks/folders in rdd format."""
    # RDD(dir_of_file_end_with_marker_origin)
    origin_dirs = list_completed_dirs(origin_prefix, marker_origin)
    # RDD(dir_of_file_end_with_marker_processed)
    processed_dirs = (list_completed_dirs(target_prefix, marker_processed)
                      # RDD(dir_of_file_end_with_marker_processed, in orgin_prefix)
                      .map(lambda path: path.replace(target_prefix, origin_prefix, 1)))
    # RDD(dir_of_to_do_tasks)
    return origin_dirs.subtract(processed_dirs)

def list_completed_dirs(prefix, marker):
    """List directories that contains COMPLETE mark up files"""
    bucket = 'apollo-platform'
    # RDD(files in prefix folders)
    return (s3_utils.list_files(bucket, prefix)
            # RDD(files_end_with_marker)
            .filter(lambda path: path.endswith(marker))
            # RDD(dirs_of_file_end_with_marker)
            .map(os.path.dirname))

def summarize_tasks(tasks, original_prefix, target_prefix):
    """Make summaries to specified tasks"""
    SummaryTuple = namedtuple('Summary', ['Task', 'Target', 'HDF5s', 'VisualPlot'])
    timestr = time.strftime("%Y%m%d-%H%M%S")
    title = 'Control Profiling Visualization Results' + ' _ %s' % timestr
    receivers = ['longtaolin@baidu.com', 'yuwang01@baidu.com', 'luoqi06@baidu.com']
    email_content = []
    for task in tasks:
        target_dir = task.replace(original_prefix, target_prefix, 1)
        email_content.append(SummaryTuple(
            Task=task,
            Target=target_dir,
            HDF5s=len(glob.glob(os.path.join(task, '*.hdf5'))),
            VisualPlot=len(glob.glob(os.path.join(target_dir, '*visualization*')))))
        file_utils.touch(os.path.join(target_dir, 'COMPLETE'))
    email_utils.send_email_info(title, email_content, receivers)

if __name__ == '__main__':
    ControlProfilingVisualization().main()
