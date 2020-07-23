#!/usr/bin/env python
import sys
sys.path.append("/fuel")

from absl import flags

from fueling.common.base_pipeline import BasePipeline

from fueling.perception.emergency_detection.train import train_yolov4

class EmergencyVehicleDetector(BasePipeline):
    """Demo pipeline."""

    def run(self):
        #train_yolov4()
        self.to_rdd(range(1)).foreach(self.train)

    @staticmethod
    def train(instance_id):
        train_yolov4()


if __name__ == '__main__':
    EmergencyVehicleDetector().main()
    #train_yolov4()
