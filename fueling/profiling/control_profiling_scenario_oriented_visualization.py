#!/usr/bin/env python

"""Visualize control features from multiple test-cases at the same scenario"""

from collections import namedtuple
import glob
import os
import tarfile
import time

import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging
import fueling.profiling.common.dir_utils as dir_utils
import fueling.profiling.feature_visualization.control_feature_scenario_oriented_visualization_utils \
       as visual_utils

class ControlProfilingScenarioOrientedVisualization(BasePipeline):
    """ Control Profiling: Visualize Control Features per Scenario"""

    def run_test(self):
        """Run test."""
        origin_prefix = '/apollo/modules/data/fuel/testdata/profiling/control_profiling/generated'
        target_prefix = origin_prefix
        # RDD(tasks), the task dirs
        todo_tasks = self.to_rdd([
            os.path.join(origin_prefix, 'ScenarioOriented_Test/Case1'),
            os.path.join(origin_prefix, 'ScenarioOriented_Test/Case2'),
        ]).cache()
        self.run(todo_tasks, origin_prefix, target_prefix)
        logging.info('Control Profiling Visualization: All Done, TEST')

    def run_prod(self):
        """Work on actual road test data. Expect a single input directory"""
        original_prefix = 'modules/control/control_profiling_hf5'
        target_prefix = original_prefix
        # RDD(tasks), the task dirs
        todo_tasks = spark_helper.cache_and_log('todo_tasks',
                                                dir_utils.get_todo_tasks(original_prefix, target_prefix,
                                                                         'COMPLETE', 'COMPLETE_PLOT'))
        self.run(todo_tasks, original_prefix, target_prefix)
        logging.info('Control Profiling Visualization: All Done, PROD')

    def run(self, todo_tasks, original_prefix, target_prefix):
        """Run the pipeline with given parameters"""
        # RDD(tasks), with absolute paths
        datalist = (todo_tasks
         # PairRDD(target_dir, task), the map of target dirs and source dirs
         .keyBy(lambda source: source.replace(original_prefix, target_prefix, 1))
         # PairRDD(target_dir, task), filter out non-existed target dirs
         .filter(spark_op.filter_key(os.path.isdir))
         # PairRDD(target_dir, hdf5_file)
         .mapValues(lambda task: glob.glob(os.path.join(task, '*.hdf5')))
         # PairRDD(target_dir, list of data_array),
         .mapValues(visual_utils.generate_segments)
         # PairRDD(target_dir, data_array), by merging the arraies in "segments" into one array
         .mapValues(visual_utils.generate_data)
         # List[(target_dir, data_array), (target_dir, data_array), ...], transform the RDD to List
         .collect())
         # List[(target_dir, data_array), (target_dir, data_array) -> Plots
        visual_utils.plot_h5_features_per_scenario(datalist)

if __name__ == '__main__':
    ControlProfilingScenarioOrientedVisualization().main()
