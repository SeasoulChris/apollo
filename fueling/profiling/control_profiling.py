#!/usr/bin/env python
"""Wrapper of vehicle calibration pipeline."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.profiling.multi_job_control_profiling_metrics import MultiJobControlProfilingMetrics
from fueling.profiling.multi_job_control_profiling_visualization import MultiJobControlProfilingVisualization


if __name__ == '__main__':
    SequentialPipeline([
        MultiJobControlProfilingMetrics(),
        MultiJobControlProfilingVisualization()
    ]).main()