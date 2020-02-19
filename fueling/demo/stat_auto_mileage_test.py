#!/usr/bin/env python
"""
A unit test demo.

Run with:
    bazel test //fueling/demo:stat_auto_mileage_test
"""
from absl.testing import absltest

from fueling.common.base_pipeline_v2 import BasePipelineTest
from fueling.demo.stat_auto_mileage import StatAutoMileage
import fueling.common.file_utils as file_utils


class StatAutoMileageTest(BasePipelineTest):
    def setUp(self):
        super().setUp(StatAutoMileage())

    def test_calculate(self):
        record = file_utils.fuel_path('fueling/demo/testdata/small.record')
        print(record)
        self.assertAlmostEqual(0.1798771794671, self.pipeline.calculate(record))


if __name__ == '__main__':
    absltest.main()
