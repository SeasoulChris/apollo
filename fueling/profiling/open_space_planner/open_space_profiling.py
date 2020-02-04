#!/usr/bin/env python
"""Wrapper of open-space profiling pipeline."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.profiling.open_space_planner.metrics import OpenSpacePlannerMetrics


if __name__ == '__main__':
    SequentialPipeline([
        OpenSpacePlannerMetrics(),
    ]).main()
