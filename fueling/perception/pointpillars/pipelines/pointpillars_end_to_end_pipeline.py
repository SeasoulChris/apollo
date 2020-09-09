#!/usr/bin/env python
"""Wrapper of daily jobs."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.perception.pointpillars.pipelines.create_kitti_data_pipeline import CreateDataKitti
from fueling.perception.pointpillars.pipelines.pointpillars_training_pipeline import (
    PointPillarsTraining,)
from fueling.perception.pointpillars.pipelines.pointpillars_export_onnx_pipeline import (
    PointPillarsExportOnnx,)

if __name__ == '__main__':
    SequentialPipeline([
        CreateDataKitti(),
        PointPillarsTraining(),
        PointPillarsExportOnnx(),
    ]).main()
