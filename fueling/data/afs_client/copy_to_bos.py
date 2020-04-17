#!/usr/bin/env python

import datetime

import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient


class CopyToBosPipeline(BasePipeline):
    """Copy AFS data to Bos"""

    def __init__(self):
        """Init"""
        # replace below as needed
        # start date
        self.start = datetime.datetime(2020, 4, 1)
        self.end = datetime.datetime(2020, 4, 1)
        # table name and message namespace
        self.afs_client = AfsClient(scan_table_name='kinglong/auto_car/cyberecord',
                                    message_namespace='kinglong/auto_car')
        # topics
        self.topics = '*'

    def run(self):
        """Run"""
        # output bos directory
        self.target_dir = self.our_storage().abs_path('modules/data/planning')
        # copy days
        interval = self.end - self.start
        # get date by start date and date increment
        def get_date(start, i): return (start + datetime.timedelta(days=i)).strftime('%Y%m%d')
        spark_helper.cache_and_log('CopyRecords',
                           # RDD(days)
                           self.to_rdd([get_date(self.start, i) for i in range(interval.days + 1)])
                           # RDD((task_id, start_time, end_time))
                           .flatMap(self.afs_client.scan_tasks).repartition(1000)
                           # RDD()
                           .map(lambda x: self.afs_client.transfer_messages(
                               x[0], x[1], x[2], self.target_dir, self.topics))
                           ).collect()

if __name__ == '__main__':
    CopyToBosPipeline().main()

