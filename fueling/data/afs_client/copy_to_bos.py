#!/usr/bin/env python

import datetime
import os

from absl import flags
import pyspark_utils.helper as spark_helper
import pyspark_utils.op as spark_op

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.data.afs_client.config as afs_config


MARKER = 'COMPLETE'

flags.DEFINE_string('start_date', '', 'copy date from what date')
flags.DEFINE_string('end_date', '', 'copy date to what date')
flags.DEFINE_string('project', 'KingLong', 'copy date for what project')
flags.DEFINE_bool('all_topics', False, 'whether copy all topics')
flags.DEFINE_string('task_id', '', 'copy some exact task id')

class CopyToBosPipeline(BasePipeline):
    """Copy AFS data to Bos"""

    def get_copy_src(self):
        """Get start and end date from input parameters"""
        task_id = self.FLAGS.get('task_id')  # something like MKZ167_20200121131624
        if task_id:
            exact_date = task_id.split('_')[1][:8]
            return exact_date, exact_date, task_id
        end_date = self.FLAGS.get('end_date') or datetime.datetime.today().strftime('%Y%m%d')
        start_date = self.FLAGS.get('start_date') or end_date
        return start_date, end_date, task_id

    def filter_tasks(self, rdd_item, project, exact_task_id, target_dir):
        """Filter the to be copied tasks"""
        task_id, map_id, region_id, task_purpose = rdd_item
        target_dir = afs_config.form_target_path(target_dir, task_id, project, map_id, task_purpose)
        logging.info(F'filtering {task_id} with {map_id}, {task_purpose}, {target_dir}')
        if exact_task_id:
            return (task_id, target_dir) if task_id == exact_task_id else None
        target_complete_file = os.path.join(target_dir, MARKER)
        if os.path.exists(target_complete_file):
            return None 
        if (not map_id in afs_config.MAP_TO_REGION or 
            not int(task_purpose) in afs_config.TASK_TO_PURPOSE):
            return None
        return (task_id, target_dir)

    def get_date(self, start_date_str, delta):
        """Get date by start date and date incremen"""
        start_date = datetime.datetime.strptime(start_date_str, '%Y%m%d').date()
        return (start_date + datetime.timedelta(days=delta)).strftime('%Y%m%d')

    def run(self):
        """Run"""
        # get input parameters
        project = self.FLAGS.get('project')
        if project not in afs_config.PROJ_TO_TABLE:
            logging.fatal(F'specified project {project} does not exist')
        task_tbl, message_tbl, keydata_tbl = afs_config.PROJ_TO_TABLE[project]
        start_date, end_date, exact_task_id = self.get_copy_src()
        interval = (datetime.datetime.strptime(end_date, '%Y%m%d').date() -
                    datetime.datetime.strptime(start_date, '%Y%m%d').date())
        afs_client = AfsClient()
        logging.info(F'copying for {task_tbl} {start_date} to {end_date} with {exact_task_id}')
        # output bos directory
        target_dir = self.our_storage().abs_path(afs_config.TARGET_PATH)
        # skip topics
        skip_topics = '' if self.FLAGS.get('all_topics') else ','.join(afs_config.SKIP_TOPICS)
        logging.info(F'copying with skipping topics {skip_topics} to {target_dir}')

        todo_tasks = spark_helper.cache_and_log('CopyPreparing',
            # RDD(days)
            self.to_rdd([self.get_date(start_date, i) for i in range(interval.days + 1)])
            # RDD((task_id, start_time, end_time))
            .flatMap(lambda x: afs_client.scan_tasks(task_tbl, x)))

        filtered_tasks = spark_helper.cache_and_log('CopyPreparing-Filter',
             todo_tasks.
             # RDD((task_id, start_time, end_time))
             map(lambda x: x[0])
             # RDD(task_id)
             .distinct()
             # RDD((task_id, capture_place, region_id, task_purpose))
             .flatMap(lambda x: afs_client.scan_keydata(keydata_tbl, x))
             # RDD((task_id, target_dir))
             .map(lambda x: self.filter_tasks(x, project, exact_task_id, target_dir))
             # RDD((task_id, target_dir))
             .filter(spark_op.not_none))

        partitions = int(os.environ.get('APOLLO_EXECUTORS', 10)) * 10
        process_tasks = spark_helper.cache_and_log('CopyProcessing', 
             todo_tasks
             # RDD(task_id, (start_time, end_time))
             .map(lambda x: (x[0], (x[1], x[2])))
             # RDD(task_id, ((start_time, end_time), target_dir))
             .join(filtered_tasks)
             # RDD(task_id, ((start_time, end_time), target_dir))
             .repartition(partitions)
             # RDD((messages))
             .map(lambda x: afs_client.transfer_messages(x, message_tbl, skip_topics))
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
