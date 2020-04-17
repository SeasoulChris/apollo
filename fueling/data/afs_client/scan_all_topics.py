#!/usr/bin/env python

import datetime
import os
import time
import pyspark_utils.helper as spark_helper
from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient

class ScanAllTopicsPipeline(BasePipeline):
    """Scan all topics and message size"""
    def run(self):
        # start date
        start = datetime.datetime(2020, 4, 1)
        # query days
        days = 1
        # get date by start date and date increment
        get_date = lambda start, i: (start + datetime.timedelta(days=i)).strftime('%Y%m%d')
        afs_client = AfsClient()

        total_topics = spark_helper.cache_and_log('ScanTopics',
            # RDD(days)
            self.to_rdd([get_date(start, i) for i in range(days)])
            # RDD((task_id, start_time, end_time))
            .flatMap(afs_client.scan_tasks).repartition(100)
            # RDD((topic, message_size))
            .flatMap(lambda x: afs_client.get_topics(x[0], x[1], x[2]))
            # RDD((topic, message_size)) reduced by topic
            .reduceByKey(lambda x, y: x + y)
            # RDD((topic, message_size)) sorted by message_size in decending order
            .sortBy(lambda x: x[1], ascending=False)
            ).collect()

        for t in total_topics:
            print(t)
        time.sleep(60 * 2)

if __name__ == '__main__':
    ScanAllTopicsPipeline().main()
