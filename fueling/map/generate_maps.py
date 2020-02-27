#!/usr/bin/env python
"""Wrapper of daily jobs."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.map.generate_base_map import MapGenSingleLine
from fueling.map.generate_local_map import LocalMapPipeline
from fueling.map.generate_sim_routing_map import SimMapPipeline


if __name__ == '__main__':
    SequentialPipeline([
        MapGenSingleLine(),
        SimMapPipeline(),
        LocalMapPipeline(),
    ]).main()
