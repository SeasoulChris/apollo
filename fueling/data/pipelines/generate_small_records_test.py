#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.data.pipelines.generate_small_records import GenerateSmallRecords


class GenerateSmallRecordsTest(BasePipelineTest):
    def setUp(self):
        super().setUp(GenerateSmallRecords())

    def test_end_to_end(self):
        self.pipeline.run_test()


if __name__ == '__main__':
    absltest.main()
