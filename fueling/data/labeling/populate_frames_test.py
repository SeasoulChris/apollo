#!/usr/bin/env python

from absl.testing import absltest

from fueling.common.base_pipeline import BasePipelineTest
from fueling.data.labeling.populate_frames import PopulateFramesPipeline


class PopulateFramesPipelineTest(BasePipelineTest):
    def setUp(self):
        super().setUp(PopulateFramesPipeline())

    def test_end_to_end(self):
        self.pipeline.run_test()

if __name__ == '__main__':
    absltest.main()
