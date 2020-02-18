from absl import flags
from absl.testing import absltest

from fueling.common.base_pipeline_v2 import BasePipelineTest
from fueling.demo.simplest_demo import SquareSum


class SimplestDemoTest(BasePipelineTest):
    def setUp(self):
        super().setUp(SquareSum())

    def test_square_sum(self):
        flags.FLAGS.square_sum_of_n = 0
        self.assertEqual(0, self.pipeline.run())
        flags.FLAGS.square_sum_of_n = 1
        self.assertEqual(1, self.pipeline.run())
        flags.FLAGS.square_sum_of_n = 10
        self.assertEqual(385, self.pipeline.run())


if __name__ == '__main__':
    absltest.main()
