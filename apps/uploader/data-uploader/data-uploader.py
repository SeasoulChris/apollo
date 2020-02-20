#!/usr/bin/env python3

"""This script upload data from vehicles to data center"""

import datetime
import os
import sys
import time
import _thread

import gflags

from bos_sync_executor import BosSyncExecutor
from road_test_listener import RoadTestTaskListener
from rsync_executor import RsyncExecutor
from serialize_job_listener import SerializeJobTaskListener
from serialize_job_executor import SerializeJobTaskExecutor
from listener import Status
from logger import Logger
import email_utils as email_utils
import global_settings as settings
import utils as utils

gflags.DEFINE_boolean('hotfix', False, 'Whether run hotfix mode')
gflags.DEFINE_boolean('email_to_all', True, 'Whether send email to everyone')


def thread_main(task, listener):
    """Main thread that connects each part and get the job done"""
    # Get src->dst list from listener's collector. src will be a tuple (src_dir, files_num, size)
    task_id = '{}_{}'.format(
        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'),
        task.replace('/', '_'))
    logger = Logger(os.path.dirname(os.path.realpath(__file__)), task_id)
    logger.log('Main Thread, executing task: {}'.format(task_id))

    receivers, title_object, executor = running_context(task_id, task, listener, logger)

    if settings.get_param(task_id).ErrorCode == settings.ErrorCode.NOT_ELIGIBLE:
        logger.log('not eligible to run, quit silently')
        listener.update_task_status(task, Status.DONE)
        return

    email_utils.send_email_info('PROCESSING{}'.format(title_object),
                                executor.get_stastic_before_executing(),
                                logger,
                                receivers)
    executor.execute(task_id, logger)
    listener.update_task_status(task, Status.DONE)
    if settings.get_param(task_id).ErrorCode == settings.ErrorCode.SUCCESS:
        email_utils.send_email_complete('COMPLETED{}'.format(title_object),
                                        executor.get_stastic_after_executing(),
                                        logger,
                                        receivers)
    else:
        email_utils.send_email_error('FAILED{}'.format(title_object),
                                     {'Error': settings.get_param(task_id).ErrorMsg},
                                     logger,
                                     receivers)
    logger.log('Main Thread, All Done with task: {}'.format(task_id))


def running_context(task_id, task, listener, logger):
    """Wrapper to get executor, receivers and other running context"""
    receivers = utils.get_recipients(gflags.FLAGS.email_to_all)
    title_object = utils.get_disk_title_object(task)
    executor = RsyncExecutor(listener.collect(task_id, task, logger))
    if isinstance(listener, SerializeJobTaskListener):
        receivers = None
        title_object = 'SerializeTaskFiles'
        executor = SerializeJobTaskExecutor(listener.collect(task_id, task, logger))
    return receivers, title_object, executor


def main():
    """Entrance."""
    gflags.FLAGS(sys.argv)
    settings.init()

    # Add new listeners here
    listeners = [
        RoadTestTaskListener(),
    ]

    heart_beat = 0
    while True:
        for listener in listeners:
            task = listener.get_available_task()
            while task:
                listener.update_task_status(task, Status.PROCESSING)
                _thread.start_new_thread(thread_main, (task, listener))
                time.sleep(1)
                task = listener.get_available_task()
        if heart_beat == 0:
            print('sleeping 10 times, each for 30 secs...')
        heart_beat = (heart_beat + 1) % 10
        time.sleep(30)


if __name__ == '__main__':
    main()
