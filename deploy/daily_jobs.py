#!/usr/bin/env python
"""Wrapper of daily jobs."""

from fueling.common.base_pipeline import SequentialPipeline
from fueling.data.pipelines.generate_small_records import GenerateSmallRecords
from fueling.data.pipelines.index_records import IndexRecords
from fueling.data.pipelines.reorg_small_records import ReorgSmallRecords
from fueling.perception.decode_video import DecodeVideoPipeline


if __name__ == '__main__':
    SequentialPipeline([
        # Record processing.
        GenerateSmallRecords(),
        ReorgSmallRecords(),
        IndexRecords(),
        DecodeVideoPipeline(),
    ]).main()
