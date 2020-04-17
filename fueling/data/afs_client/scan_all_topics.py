#!/usr/bin/env python

import datetime

import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient

class ScanAllTopicsPipeline(BasePipeline):
    """Scan all topics and message size"""

    def __init__(self):
        # replace below as needed
        # start date
        self.start = datetime.datetime(2020, 4, 1)
        self.end = datetime.datetime(2020, 4, 1)
        # table name and message namespace
        self.afs_client = AfsClient(scan_table_name='kinglong/auto_car/cyberecord',
                                    message_namespace='kinglong/auto_car')
        self.afs_client = AfsClient()

    def run(self):
        """Run"""
        # scan days
        interval = self.end - self.start
        # get date by start date and date increment
        def get_date(start, i): return (start + datetime.timedelta(days=i)).strftime('%Y%m%d')

        total_topics = spark_helper.cache_and_log('ScanTopics',
                      # RDD(days)
                      self.to_rdd([get_date(self.start, i) for i in range(interval.days)])
                      # RDD((task_id, start_time, end_time)), get task_id and time range
                      .flatMap(self.afs_client.scan_tasks).repartition(1000)
                      # PairRDD((topic, message_size)), get topics
                      .flatMap(lambda x: self.afs_client.get_topics(x[0], x[1], x[2]))
                      # PairRDD((topic, message_size)), reduced by topic
                      .reduceByKey(lambda x, y: x + y)
                      # PairRDD((topic, message_size)), sorted by message_size in decending order
                      .sortBy(lambda x: x[1], ascending=False)
                      ).collect()

        for t in total_topics:
            print(t)


if __name__ == '__main__':
    ScanAllTopicsPipeline().main()

