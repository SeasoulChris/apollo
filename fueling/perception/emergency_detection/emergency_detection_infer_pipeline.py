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

import sys
sys.path.append("/fuel")

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from yolov4.inference import inference_yolov4


class EmergencyVehicleDetector(BasePipeline):
    """Demo pipeline."""

    def run(self):
        inference_yolov4()


if __name__ == '__main__':
    EmergencyVehicleDetector().main()
