#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.data.pipelines.reorg_small_records import ReorgSmallRecords


class ReorgSmallRecordsTest(BasePipelineTest):
    def setUp(self):
        super().setUp(ReorgSmallRecords())

    def test_end_to_end(self):
        self.pipeline.run_test()


if __name__ == '__main__':
    absltest.main()
