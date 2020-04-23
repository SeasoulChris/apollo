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
flags.DEFINE_list('capture_place_list', [], 'copy task with exact comma-separated map_area_id list')
flags.DEFINE_list('region_id_list', [], 'copy task with exact comma-separated region_id list')
flags.DEFINE_list('task_purpose_list', [], 'copy task with exact comma-separated purposes list')

class CopyToBosPipeline(BasePipeline):
    """Copy AFS data to Bos"""

    def __init__(self):
        """Init"""
        # map of projects and tables/message_tb
        self.project_to_tables = {
            'KL': ('kinglong/auto_car/cyberecord',
                   'kinglong/auto_car',
                   'kinglong/auto_car/task_keydata'), # KingLong
            'MK': ('auto_car/cyberecord',
                   'auto_car',
                   'auto_car/task_keydata'), # Pilot
        }
        # topics that could be skipped
        self.skip_topics = ['PointCloud', 'camera']
        self.capture_place_list = []
        self.region_id_list = []
        self.task_purpose_list = []

    def get_copy_src(self):
        """Get start and end date from input parameters"""
        task_id = self.FLAGS.get('task_id') # something like MKZ167_20200121131624
        if task_id:
            exact_date = task_id.split('_')[1][:8]
            return exact_date, exact_date, task_id
        end_date = self.FLAGS.get('end_date') or datetime.datetime.today().strftime('%Y%m%d')
        start_date = self.FLAGS.get('start_date') or end_date
        return start_date, end_date, task_id

    def filter_tasks(self, rdd_item, task_id, target_dir, capture_place_list, region_id_list,
                     task_purpose_list):
        """Filter the to be copied tasks"""
        rdd_task_id, capture_place, region_id, task_purpose = rdd_item
        if task_id:
            return rdd_task_id == task_id
        target_complete_file = os.path.join(target_dir, rdd_task_id, MARKER)
        if os.path.exists(target_complete_file):
            return False
        if capture_place_list:
            if capture_place not in capture_place_list:
                return False
        if region_id_list:
            if region_id not in region_id_list:
                return False
        if task_purpose_list:
            if task_purpose not in task_purpose_list:
                return False
        return True

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
        task_tb, message_tb, keydata_tb = self.project_to_tables[self.FLAGS.get('project')]
        afs_client = AfsClient()
        logging.info(F'copying data for {task_tb} from {start_date} to {end_date}')
        # output bos directory
        target_dir = self.our_storage().abs_path('modules/data/planning')
        # skip topics 
        skip_topics = '' if self.FLAGS.get('all_topics') else ','.join(self.skip_topics)
        # exact map_area_id
        capture_place_list = self.FLAGS.get('capture_place_list')
        # exact region_id
        region_id_list = self.FLAGS.get('region_id_list')
        # exact task_purpose
        task_purpose_list = self.FLAGS.get('task_purpose_list')
        logging.info(F'copying with skipping topics {skip_topics} to {target_dir}')

        todo_tasks = spark_helper.cache_and_log('CopyPreparing',
            # RDD(days)
            self.to_rdd([self.get_date(start_date, i) for i in range(interval.days + 1)])
            # RDD((task_id, start_time, end_time))
            .flatMap(lambda x: afs_client.scan_tasks(task_tb, x)))

        filtered_tasks = spark_helper.cache_and_log('CopyPreparing-Filter-1',
             # RDD((task_id, start_time, end_time))
             todo_tasks.map(lambda x: x[0])
             # RDD(task_id)
             .distinct()
             # RDD((task_id, capture_place, region_id, task_purpose))
             .flatMap(lambda x: afs_client.scan_keydata(keydata_tb, x))
             # RDD((task_id, capture_place, region_id, task_purpose))
             .filter(lambda x: self.filter_tasks(x, task_id, target_dir, capture_place_list,
                                                region_id_list, task_purpose_list))
             # RDD(task_id)
             .map(lambda x: x[0])
            ).collect()

        todo_tasks = spark_helper.cache_and_log('CopyPreparing-Filter-2',
                        # RDD((task_id, start_time, end_time))
                        todo_tasks.filter(lambda x: x[0] in filtered_tasks))

        process_tasks = spark_helper.cache_and_log('CopyProcessing',
            todo_tasks
            # RDD((task_id, start_time, end_time))
            .repartition(1000)
            # RDD((messages))
            .map(lambda x: afs_client.transfer_messages(message_tb, x, target_dir, skip_topics))
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

