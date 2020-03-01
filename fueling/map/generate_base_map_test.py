#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.map.generate_base_map import MapGenSingleLine


class MapGenSingleLineTest(BasePipelineTest):
    def setUp(self):
        super().setUp(MapGenSingleLine())

    def test_end_to_end(self):
        self.pipeline.run_test()

if __name__ == '__main__':
    absltest.main()
