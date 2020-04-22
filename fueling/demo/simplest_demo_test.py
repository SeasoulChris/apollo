#!/usr/bin/env python
"""
A unit test demo.

Run with:
    bazel test //fueling/demo:simplest_demo_test
"""

from absl import flags
from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.demo.simplest_demo import SquareSum


class SimplestDemoTest(BasePipelineTest):
    def setUp(self):
        super().setUp(SquareSum())

    def test_0_or_1(self):
        flags.FLAGS.square_sum_of_n = 0
        self.assertEqual(0, self.pipeline.run())
        flags.FLAGS.square_sum_of_n = 1
        self.assertEqual(1, self.pipeline.run())

    def test_10(self):
        flags.FLAGS.square_sum_of_n = 10
        self.assertEqual(385, self.pipeline.run())


if __name__ == '__main__':
    absltest.main()
