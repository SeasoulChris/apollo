#!/usr/bin/env python
from absl import app

from fueling.data.pipelines.bag_to_record import BagToRecord
from fueling.data.pipelines.generate_small_records import GenerateSmallRecords
from fueling.data.pipelines.index_records import IndexRecords
from fueling.data.pipelines.reorg_small_records import ReorgSmallRecords


def main(argv):
    GenerateSmallRecords().__main__(argv)
    ReorgSmallRecords().__main__(argv)
    BagToRecord().__main__(argv)
    IndexRecords().__main__(argv)

if __name__ == '__main__':
    app.run(main)
