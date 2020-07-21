#!/usr/bin/env python

import io
import sys

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient
import fueling.common.logging as logging
import fueling.common.spark_helper as spark_helper


# print chinese characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ScanAllRegionNamesPipeline(BasePipeline):
    """Scan all region_id and region_name"""

    def run(self):
        """Run"""
        afs_client = AfsClient()
        total_regions = spark_helper.cache_and_log('ScanRegions', self.to_rdd([0])
                                                   # PairRDD(region_id, region_name)
                                                   .flatMap(lambda x: afs_client.scan_region_info())
                                                   # PairRDD(region_id, region_name)
                                                   .distinct()
                                                   ).collect()

        with open('/fuel/fueling/data/afs_client/total_regions.txt', 'w',
                  encoding='utf-8') as outfile:
            for region_id, region_name in total_regions:
                outfile.write(F'{region_id}\t{region_name}\n')
                logging.info(F'region_id:{region_id}\tregion_name:{region_name}')


if __name__ == '__main__':
    ScanAllRegionNamesPipeline().main()
