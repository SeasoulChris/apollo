#!/usr/bin/env python
"""
A simplest demo to calculate square sum of 1...n.

Run with:
    ./tools/submit-job-to-k8s.py --entrypoint=fueling/demo/simplest_demo.py
"""

from fueling.common.base_pipeline import BasePipeline


class SquareSum(BasePipeline):
    """Demo pipeline."""

    def __init__(self, n):
        BasePipeline.__init__(self, 'demo')
        self.n = n

    def run_test(self):
        square_sum = self.to_rdd(range(self.n)).map(lambda i: i * i).sum()
        print ('Square sum of [1, ..., {}] is {}'.format(self.n, square_sum))

    def run_prod(self):
        """For this demo, prod and test are the same."""
        return self.run_test()


if __name__ == '__main__':
    SquareSum(100).main()
