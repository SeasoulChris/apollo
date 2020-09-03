#!/usr/bin/env python
"""
Optuna optimization tuner unit test.

Run with:
    bazel test fueling/learning/autotuner/tuner:optuna_optimization_tuner_test
"""

from absl import flags
from absl.testing import absltest
import os

from fueling.common.base_pipeline import BasePipelineTest
from fueling.learning.autotuner.tuner.optuna_optimization_tuner import OptunaOptimizationTuner
import fueling.common.file_utils as file_utils


class OptunaOptimizationTunerTest(BasePipelineTest):
    TESTDATA_PATH = 'fueling/learning/autotuner/testdata/'

    def setUp(self):
        super().setUp(OptunaOptimizationTuner())

    def test_control_autotune(self):
        flags.FLAGS.cost_computation_service_url = '180.76.242.157:50052'
        flags.FLAGS.input_data_path = file_utils.fuel_path(
            F'{self.TESTDATA_PATH}control_test_config/mrac_tuner_param_config.pb.txt')
        flags.FLAGS.output_data_path = file_utils.fuel_path(
            F'{flags.FLAGS.test_tmpdir}/generated/')

        self.pipeline.init()
        self.pipeline.run()

        job_owner = self.pipeline.job_owner
        job_id = self.pipeline.job_id

        mark_file = f'{job_owner}/{job_id}/COMPLETE'
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            f'{flags.FLAGS.output_data_path}{mark_file}')))

        config_file = f'{job_owner}/{job_id}/tuner_parameters.txt'
        self.assertTrue(os.path.exists(file_utils.fuel_path(
            f'{flags.FLAGS.output_data_path}{config_file}')))

        result_file = f'{job_owner}/{job_id}/tuner_results.txt'
        output_file = file_utils.fuel_path(
            f'{flags.FLAGS.output_data_path}{result_file}')
        self.assertTrue(os.path.exists(output_file))

        output_dict = self.parse_grading_output(output_file)

        self.assertTrue('optimize_time' in output_dict)
        self.assertTrue('time_efficiency' in output_dict)
        total_time = float(output_dict['optimize_time'])
        single_step_time = float(output_dict['time_efficiency'])
        self.assertAlmostEqual(2.0, total_time / single_step_time)

        self.assertTrue('best_target' in output_dict)
        score = float(output_dict['best_target'])
        self.assertGreater(15.0, score)
        self.assertLess(0.2, score)

        self.assertTrue(
            'lat_controller_conf.steer_mrac_conf.reference_time_constant' in output_dict)
        self.assertTrue(
            'lat_controller_conf.steer_mrac_conf.adaption_matrix_p___0' in output_dict)
        param_1 = float(
            output_dict['lat_controller_conf.steer_mrac_conf.reference_time_constant'])
        param_2 = float(
            output_dict['lat_controller_conf.steer_mrac_conf.adaption_matrix_p___0'])
        self.assertGreaterEqual(0.26, param_1)
        self.assertLessEqual(0.01, param_1)
        self.assertGreaterEqual(0.1, param_2)
        self.assertLessEqual(0.0001, param_2)

    def parse_grading_output(self, output_path):
        output_dict = {}
        with open(output_path) as f:
            line = f.readline()
            while line:
                line_list = line.split(':')
                if len(line_list) > 1:
                    key = line_list[0].replace('\t', '', 1).replace('\n', '', 1)
                    val = line_list[1].replace('\t', '', 1).replace('\n', '', 1)
                    output_dict[key] = val
                line = f.readline()
        return output_dict


if __name__ == '__main__':
    absltest.main()
