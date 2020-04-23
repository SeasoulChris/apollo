#!/usr/bin/env python

import io
import sys

import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient
import fueling.common.logging as logging

# print chinese characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ScanAllAreaNamesPipeline(BasePipeline):
    """Scan all region_id and region_name"""

    def run(self):
        """Run"""
        afs_client = AfsClient()
        total_areas = spark_helper.cache_and_log('ScanAreas', self.to_rdd([0])
                                                 # PairRDD(map_area_id, map_area_name)
                                                 .flatMap(lambda x: afs_client.scan_map_area())
                                                 .distinct()
                                                 ).collect()

        with open('/fuel/fueling/data/afs_client/total_areas.txt', 'w',
                  encoding='utf-8') as outfile:
            for map_area_id, map_area_name in total_areas:
                outfile.write(F'{map_area_id}\t{map_area_name}\n')
                logging.info(F'region_id:{map_area_id}\tregion_name:{map_area_name}')


if __name__ == '__main__':
    ScanAllAreaNamesPipeline().main()

