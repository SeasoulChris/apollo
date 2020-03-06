#!/usr/bin/env python
"""
Unit test for multi_job_control_profiling_metrics.py.

Run with:
    bazel test //fueling/profiling/control:multi_job_control_profiling_metrics_test
"""

from absl import flags
from absl.testing import absltest
import os
import warnings

from fueling.common.base_pipeline import BasePipelineTest
import fueling.common.file_utils as file_utils

from fueling.profiling.control.multi_job_control_profiling_metrics \
    import MultiJobControlProfilingMetrics


class MultiJobControlProfilingMetricsTest(BasePipelineTest):
    TESTDATA_PATH = 'fueling/profiling/control/testdata/'

    def setUp(self):
        super().setUp(MultiJobControlProfilingMetrics())

    def test_run_sim_mode(self):
        flags.FLAGS.ctl_metrics_simulation_only_test = True
        flags.FLAGS.ctl_metrics_input_path = file_utils.fuel_path(F'{self.TESTDATA_PATH}sim_test/')
        flags.FLAGS.ctl_metrics_output_path = file_utils.fuel_path(F'{flags.FLAGS.test_tmpdir}/generated/')
        flags.FLAGS.ctl_metrics_todo_tasks = ''
        flags.FLAGS.ctl_metrics_conf_path = file_utils.fuel_path(F'{self.TESTDATA_PATH}conf/')
        flags.FLAGS.ctl_metrics_simulation_vehicle = 'Mkz7'
        self.pipeline.init()
        self.pipeline.run()

        output_file = ('Mkz7/Lon_Lat_Controller/sim_test/'
                       'Mkz7_Lon_Lat_Controller_control_performance_grading.txt')
        self.assertTrue(os.path.exists(file_utils.fuel_path(F'{flags.FLAGS.ctl_metrics_output_path}{output_file}')))

    def test_run_data_mode(self):
        flags.FLAGS.running_mode = 'TEST'
        flags.FLAGS.ctl_metrics_simulation_only_test = False
        flags.FLAGS.ctl_metrics_input_path = file_utils.fuel_path(F'{self.TESTDATA_PATH}road_test')
        flags.FLAGS.ctl_metrics_output_path = file_utils.fuel_path(F'{flags.FLAGS.test_tmpdir}/generated/')

        self.pipeline.init()
        self.pipeline.run()

        mark_file = 'apollo/2020/Mkz7/Lon_Lat_Controller/2019-05-01/20190501110414/COMPLETE'
        self.assertTrue(os.path.exists(file_utils.fuel_path(F'{flags.FLAGS.ctl_metrics_output_path}{mark_file}')))
        mark_file = 'apollo/2020/Transit/Lon_Lat_Controller/2019-02-25/20190225165600/COMPLETE'
        self.assertTrue(os.path.exists(file_utils.fuel_path(F'{flags.FLAGS.ctl_metrics_output_path}{mark_file}')))

        grading_file = ('apollo/2020/Mkz7/Lon_Lat_Controller/2019-05-01/20190501110414/'
                        'Mkz7_Lon_Lat_Controller_control_performance_grading.txt')
        self.assertTrue(os.path.exists(file_utils.fuel_path(F'{flags.FLAGS.ctl_metrics_output_path}{grading_file}')))
        grading_file = ('apollo/2020/Transit/Lon_Lat_Controller/2019-02-25/20190225165600/'
                        'Transit_Lon_Lat_Controller_control_performance_grading.txt')
        self.assertFalse(os.path.exists(file_utils.fuel_path(F'{flags.FLAGS.ctl_metrics_output_path}{grading_file}')))


if __name__ == '__main__':
    absltest.main(warnings='ignore')
    """
    Python 3.2 introduced ResourceWarning for unclosed system resources (network sockets, files).
    Though the code runs clean in production, there are a lot of warnings when running unit tests
    due to use of third party libraries where the warning occurs.
    """
