#!/usr/bin/env python
"""Wrapper of daily jobs."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.data.pipelines.generate_small_records import GenerateSmallRecords
from fueling.data.pipelines.index_records import IndexRecords
from fueling.data.pipelines.reorg_small_records import ReorgSmallRecords
from fueling.perception.decode_video import DecodeVideoPipeline
from fueling.profiling.control_profiling_metrics import ControlProfilingMetrics
from fueling.profiling.control_profiling_visualization import ControlProfilingVisualization


if __name__ == '__main__':
    SequentialPipeline([
        # Record processing.
        GenerateSmallRecords(),
        ReorgSmallRecords(),
        IndexRecords(),
        DecodeVideoPipeline(),
        # Control profiling.
        ControlProfilingMetrics(),
        ControlProfilingVisualization(),
    ]).main()
