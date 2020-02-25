#!/usr/bin/env python
"""Wrapper of vehicle calibration pipeline."""

from fueling.common.base_pipeline_v2 import SequentialPipelineV2
from fueling.profiling.control.multi_job_control_profiling_metrics \
    import MultiJobControlProfilingMetrics
from fueling.profiling.control.multi_job_control_profiling_visualization \
    import MultiJobControlProfilingVisualization


if __name__ == '__main__':
    SequentialPipelineV2([
        MultiJobControlProfilingMetrics(),
        MultiJobControlProfilingVisualization(),
    ]).main()
