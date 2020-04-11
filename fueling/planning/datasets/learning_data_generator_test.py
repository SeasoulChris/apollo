#!/usr/bin/env python

from absl import logging
from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.planning.datasets.learning_data_generator import LearningDataGenerator

class LearningDataGeneratorTest(BasePipelineTest):
    def setUp(self):
        super().setUp(LearningDataGenerator())

    def test_learning_data_generator(self):
        """test"""
        # self.assertEqual(0, self.pipeline.run_test())


if __name__ == '__main__':
    absltest.main()
