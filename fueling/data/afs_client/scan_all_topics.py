#!/usr/bin/env python

import datetime

from absl import flags
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient
import fueling.common.logging as logging

flags.DEFINE_string('start_date', '', 'copy date from what date')
flags.DEFINE_string('end_date', '', 'copy date to what date')
flags.DEFINE_string('project', 'KL', 'topics for what project')


class ScanAllTopicsPipeline(BasePipeline):
    """Scan all topics and message size"""

    def __init__(self):
        self.project_to_tables = {
            'KL': ('kinglong/auto_car/cyberecord',
                   'kinglong/auto_car'),  # KingLong
            'MK': ('auto_car/cyberecord',
                   'auto_car'),  # Pilot
        }

    def get_date(self, start_date_str, delta):
        """Get date by start date and date incremen"""
        start_date = datetime.datetime.strptime(start_date_str, '%Y%m%d').date()
        return (start_date + datetime.timedelta(days=delta)).strftime('%Y%m%d')

    def run(self):
        """Run"""
        end_date = self.FLAGS.get('end_date') or datetime.datetime.today().strftime('%Y%m%d')
        start_date = self.FLAGS.get('start_date') or end_date
        # scan days
        interval = (datetime.datetime.strptime(end_date, '%Y%m%d').date()
                    - datetime.datetime.strptime(start_date, '%Y%m%d').date())
        task_tb, message_tb = self.project_to_tables[self.FLAGS.get('project')]
        afs_client = AfsClient()
        total_topics = spark_helper.cache_and_log(
            'ScanTopics',
            # RDD(days)
            self.to_rdd([self.get_date(start_date, i) for i in range(interval.days + 1)])
            # RDD((task_id, start_time, end_time)), get
            # task_id and time range
            .flatMap(lambda x: afs_client.scan_tasks(task_tb, x)).repartition(1000)
            # PairRDD((topic, message_size)), get topics
            .flatMap(lambda x: afs_client.get_topics(message_tb, x[0], x[1], x[2]))
            # PairRDD((topic, message_size)), reduced by topic
            .reduceByKey(lambda x, y: x + y)
            # PairRDD((topic, message_size)), sorted by
            # message_size in decending order
            .sortBy(lambda x: x[1], ascending=False)).collect()
        with open('/fuel/fueling/data/afs_client/data/total_topics.txt', 'w',
                  encoding='utf-8') as outfile:
            for topic, message_size in total_topics:
                outfile.write(F'{topic}\t{message_size}\n')
                logging.info(F'topic:{topic},message_size:{message_size}')


if __name__ == '__main__':
    ScanAllTopicsPipeline().main()
