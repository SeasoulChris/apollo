#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.map.generate_sim_routing_map import SimMapPipeline


class SimMapPipelineTest(BasePipelineTest):
    def setUp(self):
        super().setUp(SimMapPipeline())

    def test_end_to_end(self):
        self.pipeline.run_test()


if __name__ == '__main__':
    absltest.main()
