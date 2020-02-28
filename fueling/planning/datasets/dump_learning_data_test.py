#!/usr/bin/env python

from absl import logging
from absl.testing import absltest

from fueling.common.base_pipeline_v2 import BasePipelineTest
from fueling.planning.datasets.dump_learning_data import DumpLearningData

class DumpLearningDataTest(BasePipelineTest):
    def setUp(self):
        super().setUp(DumpLearningData())

    def test_dump_learning_data(self):
        """test"""
        # self.assertEqual(0, self.pipeline.run_test())

if __name__ == '__main__':
    absltest.main()
