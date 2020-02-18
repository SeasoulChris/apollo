#!/usr/bin/env python
"""
A simplest demo to calculate square sum of 1...n.

Run at local:
    bazel run //fueling/demo:simplest_demo
    bazel run //fueling/demo:simplest_demo -- --square_sum_of_n=1000

Run in cloud:
    bazel run //fueling/demo:simplest_demo -- --cloud
    bazel run //fueling/demo:simplest_demo -- --cloud --square_sum_of_n=1000
"""

from absl import flags

from fueling.common.base_pipeline_v2 import BasePipelineV2


flags.DEFINE_integer('square_sum_of_n', 100, 'Square sum of n: (1^2 + ... + n^2)')


class SquareSum(BasePipelineV2):
    """Demo pipeline."""
    def run(self):
        """Calculate (1^2 + ... + n^2)."""
        n = flags.FLAGS.square_sum_of_n
        square_sum = self.to_rdd(range(1, n + 1)).map(lambda i: i * i).sum()
        print ('Square sum of [1, ..., {}] is {}'.format(n, square_sum))
        return square_sum


if __name__ == '__main__':
    SquareSum().main()
