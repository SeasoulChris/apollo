#!/usr/bin/env python
"""Wrapper of daily jobs."""

from absl import app

from fueling.data.pipelines.generate_small_records import GenerateSmallRecords
from fueling.data.pipelines.index_records import IndexRecords
from fueling.data.pipelines.reorg_small_records import ReorgSmallRecords
from fueling.perception.decode_video import DecodeVideoPipeline


def main(argv):
    GenerateSmallRecords().__main__(argv)
    ReorgSmallRecords().__main__(argv)
    IndexRecords().__main__(argv)
    DecodeVideoPipeline().__main__(argv)


if __name__ == '__main__':
    app.run(main)
