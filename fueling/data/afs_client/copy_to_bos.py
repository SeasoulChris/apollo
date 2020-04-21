#!/usr/bin/env python

import datetime
import os

from absl import flags
import pyspark_utils.helper as spark_helper

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


MARKER = 'COMPLETE'

flags.DEFINE_string('start_date', '', 'copy date from what date')
flags.DEFINE_string('end_date', '', 'copy date to what date')
flags.DEFINE_string('project', 'KL', 'copy date for what project')
flags.DEFINE_bool('all_topics', False, 'whether copy all topics')
flags.DEFINE_string('task_id', '', 'copy some exact task id')

class CopyToBosPipeline(BasePipeline):
    """Copy AFS data to Bos"""

    def __init__(self):
        """Init"""
        # map of projects and tables/namespace
        self.project_to_tables = {
            'KL': ('kinglong/auto_car/cyberecord', 'kinglong/auto_car'), # KingLong
            'MK': ('/auto_car/cyberecord', 'auto_car'), # Pilot
        }
        # topics that could be skipped
        self.skip_topics = ['PointCloud', 'camera']

    def get_copy_src(self):
        """Get start and end date from input parameters"""
        task_id = self.FLAGS.get('task_id') # something like MKZ167_20200121131624
        if task_id:
            exact_date = task_id.split('_')[1][:8]
            return exact_date, exact_date, task_id
        end_date = self.FLAGS.get('end_date') or datetime.datetime.today().strftime('%Y%m%d')
        start_date = self.FLAGS.get('start_date') or end_date
        return start_date, end_date, task_id

    def filter_tasks(self, rdd_item, task_id, target_dir):
        """Filter the to be copied tasks"""
        rdd_task_id, start_time, end_time = rdd_item 
        if task_id:
            return rdd_task_id == task_id 
        target_complete_file = os.path.join(target_dir, rdd_task_id, MARKER)
        return not os.path.exists(target_complete_file)
        
    def get_date(self, start_date_str, delta):
        """Get date by start date and date incremen"""
        start_date = datetime.datetime.strptime(start_date_str, '%Y%m%d').date()
        return (start_date + datetime.timedelta(days=delta)).strftime('%Y%m%d')
        
    def run(self):
        """Run"""
        # get input parameters
        start_date, end_date, task_id = self.get_copy_src()
        interval = (datetime.datetime.strptime(end_date, '%Y%m%d').date() - 
                    datetime.datetime.strptime(start_date, '%Y%m%d').date())
        table_name, namespace = self.project_to_tables[self.FLAGS.get('project')]
        afs_client = AfsClient(table_name, namespace)
        logging.info(F'copying data for {table_name} from {start_date} to {end_date}')
        # output bos directory
        target_dir = self.our_storage().abs_path('modules/data/planning')
        # skip topics 
        skip_topics = '' if self.FLAGS.get('all_topics') else ','.join(self.skip_topics)
        logging.info(F'copying with skipping topics {skip_topics} to {target_dir}')

        todo_tasks = spark_helper.cache_and_log('CopyPreparing',
            # RDD(days)
            self.to_rdd([self.get_date(start_date, i) for i in range(interval.days + 1)])
            # RDD((task_id, start_time, end_time))
            .flatMap(afs_client.scan_tasks)
            # RDD((task_id, start_time, end_time))
            .filter(lambda x: self.filter_tasks(x, task_id, target_dir)))

        process_tasks = spark_helper.cache_and_log('CopyProcessing',
            todo_tasks
            # RDD((task_id, start_time, end_time))
            .repartition(1000)
            # RDD((messages))
            .map(lambda x: afs_client.transfer_messages(x, target_dir, skip_topics))
            # RDD(completed dirs)
            .map(os.path.dirname)
            # RDD(completed dirs)
            .distinct())

        (process_tasks
            # RDD(target_dir/COMPLETE)
            .map(lambda target_dir: os.path.join(target_dir, MARKER))
            # Make target_dir/COMPLETE files.
            .foreach(file_utils.touch))

if __name__ == '__main__':
    CopyToBosPipeline().main()

