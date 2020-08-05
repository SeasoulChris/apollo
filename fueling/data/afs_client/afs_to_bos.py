#!/usr/bin/env python

import collections
import datetime
import os

from absl import flags

from fueling.common.base_pipeline import BasePipeline
from fueling.data.afs_client.client import AfsClient
import fueling.common.email_utils as email_utils
import fueling.common.file_utils as file_utils
import fueling.common.logging as logging
import fueling.common.spark_helper as spark_helper
import fueling.common.spark_op as spark_op
import fueling.data.afs_client.config as afs_config


MARKER = 'COMPLETE'

flags.DEFINE_string('start_date', '', 'copy date from what date')
flags.DEFINE_string('end_date', '', 'copy date to what date')
flags.DEFINE_string('project', 'KingLong', 'copy date for what project')
flags.DEFINE_bool('all_topics', False, 'whether copy all topics')
flags.DEFINE_string('task_id', '', 'copy some exact task id')
flags.DEFINE_bool('download_logs', False, 'whether download logs')


class AfsToBosPipeline(BasePipeline):

    """Copy AFS data to Bos"""

    def get_copy_src(self):
        """Get start and end date from input parameters"""
        task_id = self.FLAGS.get('task_id')  # something like MKZ167_20200121131624
        if task_id:
            exact_date = task_id.split('_')[1][:8]
            return exact_date, exact_date, task_id
        # By default get two days data to avoid some being missed due to time diff
        today = datetime.datetime.today()
        yesterday = today - datetime.timedelta(days=1)
        end_date = self.FLAGS.get('end_date') or today.strftime('%Y%m%d')
        start_date = self.FLAGS.get('start_date') or yesterday.strftime('%Y%m%d')
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
        if (map_id not in afs_config.MAP_TO_REGION
                or int(task_purpose) not in afs_config.TASK_TO_PURPOSE):
            return None
        return (task_id, target_dir)

    def get_date(self, start_date_str, delta):
        """Get date by start date and date incremen"""
        start_date = datetime.datetime.strptime(start_date_str, '%Y%m%d').date()
        return (start_date + datetime.timedelta(days=delta)).strftime('%Y%m%d')

    def send_summary_email(self, completed_dirs):
        """Send email notification"""
        if not completed_dirs:
            logging.info('No need to send summary for empty result')
            return
        SummaryTuple = collections.namedtuple('Summary', ['TaskDirectory'])
        title = F'Transfered AFS data for {len(completed_dirs)} tasks'
        message = [SummaryTuple(TaskDirectory=task_dir) for task_dir in completed_dirs]

        try:
            email_utils.send_email_info(title, message, email_utils.DATA_TEAM)
        except Exception as error:
            logging.error('Failed to send summary: {}'.format(error))

    def run(self):
        """Run"""
        # get input parameters
        project = self.FLAGS.get('project')
        if project not in afs_config.PROJ_TO_TABLE:
            logging.fatal(F'specified project {project} does not exist')
        task_tbl, message_tbl, keydata_tbl, log_tbl = afs_config.PROJ_TO_TABLE[project]
        start_date, end_date, exact_task_id = self.get_copy_src()
        interval = (datetime.datetime.strptime(end_date, '%Y%m%d').date()
                    - datetime.datetime.strptime(start_date, '%Y%m%d').date())
        afs_client = AfsClient()
        logging.info(F'copying for {task_tbl} {start_date} to {end_date} with {exact_task_id}')
        # output bos directory
        target_dir = self.our_storage().abs_path(afs_config.TARGET_PATH)
        # topics and skip topics
        topics = '*' if self.FLAGS.get('all_topics') else ','.join(afs_config.TOPICS)
        skip_topics = '' if self.FLAGS.get('all_topics') else ','.join(afs_config.SKIP_TOPICS)
        logging.info(F'copying with topics {topics}, skipping topics {skip_topics} to {target_dir}')

        todo_tasks = spark_helper.cache_and_log(
            'TodoTasks',
            # RDD(days)
            self.to_rdd([self.get_date(start_date, i) for i in range(interval.days + 1)])
            # PairRDD(task_id, start_time, end_time)
            .flatMap(lambda x: afs_client.scan_tasks(task_tbl, x)))

        tasks_key_data = spark_helper.cache_and_log(
            'Tasks-Keydata',
            # PairRDD(task_id, start_time, end_time)
            todo_tasks
            # RDD(task_id)
            .map(lambda x: x[0])
            # RDD(task_id)
            .distinct()
            # PairRDD(task_id, capture_place, region_id, task_purpose)
            .flatMap(lambda x: afs_client.scan_keydata(keydata_tbl, x)))

        filtered_tasks = spark_helper.cache_and_log(
            'FilteredTasks',
            tasks_key_data 
            # PairRDD(task_id, target_dir)
            .map(lambda x: self.filter_tasks(x, project, exact_task_id, target_dir))
            # PairRDD(task_id, target_dir)
            .filter(spark_op.not_none))

        partitions = int(os.environ.get('APOLLO_EXECUTORS', 10)) * 10
        process_tasks = spark_helper.cache_and_log(
            'ProcessTasks',
            todo_tasks
            # PairRDD(task_id, (start_time, end_time))
            .map(lambda x: (x[0], (x[1], x[2])))
            # PairRDD(task_id, ((start_time, end_time), target_dir))
            .join(filtered_tasks)
            # PairRDD(task_id, ((start_time, end_time), target_dir))
            .repartition(partitions)
            # RDD((messages))
            .map(lambda x: afs_client.transfer_messages(x, message_tbl, skip_topics, topics))
            # RDD(completed dirs)
            .map(os.path.dirname)
            # RDD(completed dirs)
            .distinct())

        (process_tasks
            # RDD(target_dir/COMPLETE)
            .map(lambda target_dir: os.path.join(target_dir, MARKER))
            # Make target_dir/COMPLETE files.
            .foreach(file_utils.touch))

        # Download logs at last if necessary
        if self.FLAGS.get('download_logs') and afs_config.LOG_NAMES:
            # PairRDD(task_id, target_dir)
            (filtered_tasks
                # PairRDD(task_id, log_dir)
                .mapValues(lambda target_dir: target_dir.replace(
                    afs_config.TARGET_PATH, afs_config.LOG_PATH, 1))
                # PairRDD(task_id, log_dir)
                .foreach(lambda task_target: afs_client.get_logs(
                    log_tbl, task_target, ','.join(afs_config.LOG_NAMES))))

        self.send_summary_email(process_tasks.collect())


if __name__ == '__main__':
    AfsToBosPipeline().main()
