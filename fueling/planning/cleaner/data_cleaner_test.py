#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.planning.cleaner.data_cleaner import CleanPlanningRecords


class GenerateSmallRecordsTest(BasePipelineTest):
    def setUp(self):
        super().setUp(CleanPlanningRecords())

    def test_end_to_end(self):
        self.pipeline.run_test()


if __name__ == '__main__':
    absltest.main()
