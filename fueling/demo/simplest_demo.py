#!/usr/bin/env python
"""
A simplest demo to calculate square sum of 1...n.

Run with:
    ./tools/submit-job-to-k8s.py --main=fueling/demo/simplest_demo.py

V2:
Run at local:
    bazel run //fueling/demo:simplest_demo
Run in cloud:
    ./v2/cloud_run.py --main=bazel-bin/fueling/demo/simplest_demo.zip
"""

from fueling.common.base_pipeline import BasePipeline


class SquareSum(BasePipeline):
    """Demo pipeline."""

    def __init__(self, n):
        self.n = n

    def run_test(self):
        square_sum = self.to_rdd(range(self.n)).map(lambda i: i * i).sum()
        print ('Square sum of [1, ..., {}] is {}'.format(self.n, square_sum))

    def run_prod(self):
        """For this demo, prod and test are the same."""
        return self.run_test()


if __name__ == '__main__':
    SquareSum(100).main()
