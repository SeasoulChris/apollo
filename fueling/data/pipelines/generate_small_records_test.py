#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.data.pipelines.generate_small_records import GenerateSmallRecords
import fueling.common.file_utils as file_utils


class GenerateSmallRecordsTest(BasePipelineTest):
    def setUp(self):
        super().setUp(GenerateSmallRecords())

    def test_end_to_end(self):
        todo_records = self.pipeline.to_rdd(
            [file_utils.fuel_path('fueling/demo/testdata/small.record')])
        src_prefix = file_utils.fuel_path('fueling/demo/testdata')
        dst_prefix = self.pipeline.FLAGS.get('test_tmpdir')
        self.pipeline.run_internal(todo_records, src_prefix, dst_prefix)


if __name__ == '__main__':
    absltest.main()
