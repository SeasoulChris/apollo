#!/usr/bin/env python
"""Wrapper of daily jobs."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.data.pipelines.generate_small_records import GenerateSmallRecords
from fueling.data.pipelines.index_records import IndexRecords
from fueling.data.pipelines.reorg_small_records import ReorgSmallRecords
from fueling.perception.decode_video import DecodeVideoPipeline
from fueling.profiling.reorg_smallrecords_by_vehicle import ReorgSmallRecordsByVehicle
from fueling.profiling.multi_job_control_profiling_metrics import MultiJobControlProfilingMetrics
from fueling.profiling.multi_job_control_profiling_visualization import MultiJobControlProfilingVisualization


if __name__ == '__main__':
    SequentialPipeline([
        # Record processing.
        GenerateSmallRecords(),
        ReorgSmallRecords(),
        IndexRecords(),
        DecodeVideoPipeline(),
        # Control profiling.
        ReorgSmallRecordsByVehicle(),
        # TODO(uncomment when get a smooth way pass the input_data_path reorgized by last job):
        # MultiJobControlProfilingMetrics(),
        # MultiJobControlProfilingVisualization(),
    ]).main()
