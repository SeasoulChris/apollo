#!/usr/bin/env python
"""
Unit test for metrics.py.

Run with:
    bazel test //fueling/profiling/open_space_planner:metrics_test
"""

from absl import flags
import os
import shutil

from fueling.common.base_pipeline import BasePipelineTest
import fueling.common.file_utils as file_utils

from fueling.profiling.open_space_planner.metrics import OpenSpacePlannerMetrics


class OpenSpacePlannerMetricsTest(BasePipelineTest):
    TESTDATA_PATH = 'fueling/profiling/open_space_planner/testdata'

    def setUp(self):
        super().setUp(OpenSpacePlannerMetrics())

    def test_no_obstacle_with_report(self):
        flags.FLAGS.input_data_path = file_utils.fuel_path(
            F'{self.TESTDATA_PATH}/no_obstacle')
        flags.FLAGS.output_data_path = file_utils.fuel_path(
            F'{flags.FLAGS.test_tmpdir}/generated')
        flags.FLAGS.open_space_planner_profiling_generate_report = True
        flags.FLAGS.open_space_planner_profiling_debug = True
        flags.FLAGS.open_space_planner_profiling_simulation_only = True

        self.pipeline.init()
        self.pipeline.run()

        grading_output = file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/open_space_performance_grading.txt')
        self.assertTrue(os.path.exists(grading_output))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/open_space_performance_grading.json')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/__open_space_performance_stats.json')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/__open_space_feature_data.json')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/stage_feature.hdf5')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/latency_feature.hdf5')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/zigzag_feature.hdf5')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/trajectory_feature.hdf5')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/feature_visualization.pdf')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/feature_timeline_visualization.pdf')))

        output_dict = self.parse_grading_output(grading_output)
        self.assertEqual(21, len(output_dict))
        self.assertListEqual(['N.A.', '1260.302', '1260.302', '0.000',
                              '1260.302', 'N.A.', '195'], output_dict['zigzag_time'])
        self.assertListEqual(['14950.000001', '1'], output_dict['stage_completion_time'])
        self.assertListEqual(['0', '15.570%', '2.347%', '3.938%', '11.645%',
                              'N.A.', '26576'], output_dict['lateral_negative_jerk_ratio'])
        self.assertListEqual(['0', '0.000%', '0.000%', '0.000%', '0.000%',
                              'N.A.', '0'], output_dict['distance_to_obstacles_ratio'])
        shutil.rmtree(flags.FLAGS.output_data_path)

    def test_obstacle_no_report(self):
        flags.FLAGS.input_data_path = file_utils.fuel_path(
            F'{self.TESTDATA_PATH}/obstacle')
        flags.FLAGS.output_data_path = file_utils.fuel_path(
            F'{flags.FLAGS.test_tmpdir}/generated')
        flags.FLAGS.open_space_planner_profiling_generate_report = False
        flags.FLAGS.open_space_planner_profiling_debug = False

        self.pipeline.init()
        result_data = self.pipeline.run()

        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/open_space_performance_grading.txt')))
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/open_space_performance_grading.json')))
        self.assertFalse(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/stage_feature.hdf5')))
        self.assertFalse(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/latency_feature.hdf5')))
        self.assertFalse(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/zigzag_feature.hdf5')))
        self.assertFalse(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/trajectory_feature.hdf5')))
        self.assertFalse(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/feature_visualization.pdf')))
        self.assertFalse(os.path.exists(file_utils.fuel_path(
            F'{flags.FLAGS.output_data_path}/feature_timeline_visualization.pdf')))

        self.assertEqual(1, len(result_data))
        _, output = result_data[0]
        self.assertEqual(20, len(output))
        self.assertListEqual([0.005628517823639775, 1.0, 0.3183317924567638, 0.13117638380055793,
                              0.42628587217711833, 533], output.non_gear_switch_length_ratio)
        self.assertTupleEqual((4.141625030490488e-05, 1), output.initial_heading_diff_ratio)
        self.assertListEqual([0.0, 0.33018652986519514, 0.1823102366181574, 0.06817409352324616,
                              0.33018652953932903, 9053], output.distance_to_obstacles_ratio)
        shutil.rmtree(flags.FLAGS.output_data_path)

    def parse_grading_output(self, output_path):
        output_dict = {}
        with open(output_path) as f:
            line = f.readline()
            while line:
                line_list = line.split()
                output_dict[line_list[0]] = line_list[1:]
                line = f.readline()
        return output_dict


if __name__ == '__main__':
    # TODO(xiaoxq): Fix.
    # absltest.main(warnings='ignore')
    """
    Python 3.2 introduced ResourceWarning for unclosed system resources (network sockets, files).
    Though the code runs clean in production, there are a lot of warnings when running unit tests
    due to use of third party libraries where the warning occurs.
    """
