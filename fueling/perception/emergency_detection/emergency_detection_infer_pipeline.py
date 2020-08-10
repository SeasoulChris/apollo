#!/usr/bin/env python

from yolov4.inference import inference_yolov4

from fueling.common.base_pipeline import BasePipeline


class EmergencyVehicleDetector(BasePipeline):
    """Demo pipeline."""

    def run(self):
        inference_yolov4()


if __name__ == '__main__':
    EmergencyVehicleDetector().main()
    # inference_yolov4(is_local=True)
