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
        flags.FLAGS.input_data_path = file_utils.fuel_path(F'{self.TESTDATA_PATH}sim_test/')
        flags.FLAGS.output_data_path = file_utils.fuel_path(
            F'{flags.FLAGS.test_tmpdir}/generated/')
        flags.FLAGS.ctl_metrics_todo_tasks = ''
        flags.FLAGS.ctl_metrics_conf_path = file_utils.fuel_path(F'{self.TESTDATA_PATH}conf/')
        flags.FLAGS.ctl_metrics_simulation_vehicle = 'Mkz7'
        self.pipeline.init()
        self.pipeline.run()

        output_file = ('Mkz7/Lon_Lat_Controller/sim_test/'
                       'Mkz7_Lon_Lat_Controller_control_performance_grading.txt')
        output_path = file_utils.fuel_path(F'{flags.FLAGS.output_data_path}{output_file}')
        self.assertTrue(os.path.exists(output_path))

        output_dict = self.parse_grading_output(output_path)
        self.assertEqual(44, len(output_dict))
        self.assertTrue('station_err_std' in output_dict)
        self.assertListEqual(['17.555%', '5102'], output_dict['station_err_std'])
        self.assertTrue('speed_err_peak' in output_dict)
        self.assertListEqual(['105.801%', '5218', '22.390'], output_dict['speed_err_peak'])
        self.assertTrue('jerk_bad_sensation' in output_dict)
        self.assertListEqual(['2.683%', '5218'], output_dict['jerk_bad_sensation'])
        self.assertTrue('throttle_control_usage' in output_dict)
        self.assertListEqual(['20.717%', '5218'], output_dict['throttle_control_usage'])
        # ToDo(Yu): clarify why some other testers reported four different metrics with
        # sim_records: throttle_deadzone_mean, brake_deadzone_mean, pose_heading_offset_std,
        # and pose_heading_offset_peak
        # self.assertTrue('brake_deadzone_mean' in output_dict)
        # self.assertListEqual(['5.355%', '2889'], output_dict['brake_deadzone_mean'])
        self.assertTrue('control_error_code_count' in output_dict)
        self.assertListEqual(['0.000%', '5218'], output_dict['control_error_code_count'])
        self.assertTrue('weighted_score' in output_dict)
        self.assertListEqual(['18.736%', '5218'], output_dict['weighted_score'])

    def test_run_data_mode(self):
        flags.FLAGS.running_mode = 'TEST'
        flags.FLAGS.ctl_metrics_simulation_only_test = False
        flags.FLAGS.input_data_path = file_utils.fuel_path(F'{self.TESTDATA_PATH}road_test')
        flags.FLAGS.output_data_path = file_utils.fuel_path(
            F'{flags.FLAGS.test_tmpdir}/generated/')

        self.pipeline.init()
        self.pipeline.run()

        mark_file = 'apollo/2020/Mkz7/Lon_Lat_Controller/2019-05-01/20190501110414/COMPLETE'
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}{mark_file}')))
        mark_file = 'apollo/2020/Transit/Lon_Lat_Controller/2019-02-25/20190225165600/COMPLETE'
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}{mark_file}')))

        grading_file = ('apollo/2020/Mkz7/Lon_Lat_Controller/2019-05-01/20190501110414/'
                        'Mkz7_Lon_Lat_Controller_control_performance_grading.txt')
        output_path = file_utils.fuel_path(F'{flags.FLAGS.output_data_path}{grading_file}')
        self.assertTrue(os.path.exists(output_path))

        output_dict = self.parse_grading_output(output_path)
        self.assertEqual(44, len(output_dict))
        self.assertTrue('ending_station_err_trajectory_0' in output_dict)
        self.assertListEqual(['34.425%', '177', '1556733975.015'],
                             output_dict['ending_station_err_trajectory_0'])
        self.assertTrue('weighted_score' in output_dict)
        self.assertListEqual(['22.928%', '5399'], output_dict['weighted_score'])

        grading_file = ('apollo/2020/Transit/Lon_Lat_Controller/2019-02-25/20190225165600/'
                        'Transit_Lon_Lat_Controller_control_performance_grading.txt')
        self.assertFalse(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}{grading_file}')))

    def parse_grading_output(self, output_path):
        output_dict = {}
        with open(output_path) as f:
            line = f.readline()
            while line:
                line_list = line.split()
                output_dict[line_list[1]] = line_list[2:]
                line = f.readline()
        return output_dict


if __name__ == '__main__':
    absltest.main(warnings='ignore')
    """
    Python 3.2 introduced ResourceWarning for unclosed system resources (network sockets, files).
    Though the code runs clean in production, there are a lot of warnings when running unit tests
    due to use of third party libraries where the warning occurs.
    """
