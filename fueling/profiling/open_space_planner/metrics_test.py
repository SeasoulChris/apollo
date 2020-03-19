#!/usr/bin/env python
"""
Unit test for metrics.py.

Run with:
    bazel test //fueling/profiling/open_space_planner:metrics_test
"""

from absl import flags
from absl.testing import absltest
import os
import warnings

from fueling.common.base_pipeline import BasePipelineTest
import fueling.common.file_utils as file_utils

from fueling.profiling.open_space_planner.metrics import OpenSpacePlannerMetrics


class OpenSpacePlannerMetricsTest(BasePipelineTest):
    TESTDATA_PATH = 'fueling/profiling/open_space_planner/testdata'

    def setUp(self):
        super().setUp(OpenSpacePlannerMetrics())

    def test_run(self):
        flags.FLAGS.running_mode = 'TEST'
        flags.FLAGS.open_space_planner_profiling_input_path = file_utils.fuel_path(self.TESTDATA_PATH)
        flags.FLAGS.open_space_planner_profiling_output_path = file_utils.fuel_path(F'{flags.FLAGS.test_tmpdir}/generated')

        self.pipeline.init()
        self.pipeline.run()

        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.open_space_planner_profiling_output_path}/_00000.hdf5')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.open_space_planner_profiling_output_path}/open_space_performance_grading.txt')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.open_space_planner_profiling_output_path}/visualization.pdf')))


if __name__ == '__main__':
    absltest.main(warnings='ignore')
    """
    Python 3.2 introduced ResourceWarning for unclosed system resources (network sockets, files).
    Though the code runs clean in production, there are a lot of warnings when running unit tests
    due to use of third party libraries where the warning occurs.
    """
