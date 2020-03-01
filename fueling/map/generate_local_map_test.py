#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.map.generate_local_map import LocalMapPipeline


class LocalMapPipelineTest(BasePipelineTest):
    def setUp(self):
        super().setUp(LocalMapPipeline())

    def test_end_to_end(self):
        self.pipeline.run_test()

if __name__ == '__main__':
    absltest.main()
