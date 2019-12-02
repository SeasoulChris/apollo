#!/usr/bin/env python
"""Wrapper of perception model training pipeline jobs"""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.perception.YOLOv3.yolov3_evaluate import Yolov3Evaluate
from fueling.perception.YOLOv3.yolov3_inference import Yolov3Inference
from fueling.perception.YOLOv3.yolov3_training import Yolov3Training


if __name__ == '__main__':
    SequentialPipeline([
        Yolov3Training(),
        Yolov3Inference(),
        Yolov3Evaluate(),
    ]).main()
