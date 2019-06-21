#!/usr/bin/env python3

"""Listen to the interested events and update status when those happen"""

from enum import Enum
import os

import global_settings as settings
import utils as utils

class Status(Enum):
    """Status enum."""
    UNKNOWN = 1
    INITIAL = 2
    PROCESSING = 3
    DONE = 4

class Listener(object):
    """Base class of listeners."""
    def __init__(self):
        # Map of tasks and their status
        self._tasks = {}

    def get_available_task(self):
        """Check if there are any tasks that are ready to be uploaded"""
        for task in self._tasks:
            if self._tasks[task] == Status.INITIAL:
                return task
        return None

    def update_task_status(self, task, status):
        """Update status for particular task"""
        self._tasks[task] = status

    def collect(self, task_id, task, logger):
        """Collector, collects the src and dst as well as their related information"""
        raise Exception('collect is Not Implemented for base class')

class DiskManagement(object):
    @staticmethod
    def mount(task_id, disk_label, logger):
        """Mount the disk"""
        if not DiskManagement._is_eligible_to_mount(disk_label):
            logger.log('disk not eligible to mount: {}'.format(disk_label))
            settings.set_param(
                task_id,
                settings.get_param(task_id)._replace(ErrorCode=settings.ErrorCode.NOT_ELIGIBLE))
            return None
        DiskManagement.unmount(disk_label)
        mount_point = DiskManagement._get_mount_point(disk_label)
        utils.makedirs(mount_point)
        mount_cmd = 'mount {} {}'.format(disk_label, mount_point)
        if os.system(mount_cmd) != 0:
            error_msg = 'failed to execute mount command: {}'.format(mount_cmd)
            logger.log(error_msg)
            settings.set_param(
                task_id,
                settings.get_param(task_id)._replace(ErrorCode=settings.ErrorCode.FAIL,
                                                     ErrorMsg=error_msg))
            return None
        return mount_point

    @staticmethod
    def unmount(disk_label):
        """Unmount the disk"""
        os.system('umount -l {}'.format(disk_label))
    
    @staticmethod
    def is_mounted(disk_label):
        """Check if already mounted"""
        valid_mount_cols = 6
        mount_info = utils.check_output('lsblk | grep {}'.format(disk_label.split('/')[-1 :]))
        return mount_info and len(mount_info[0].split()) > valid_mount_cols

    @staticmethod
    def _get_mount_point(disk_label):
        """Get mount point based on disk_label name"""
        str_all = 'abcdefghijklmnopqrstuvwxyz'
        num_all = '0123456789'
        mount_root = '/media/apollo/apollo'
        num = disk_label[-1:]
        if num >= '0' and num <= '9':
            dev = disk_label[-2:-1]
        else:
            dev = num
            num = '0'
        mount_point = '{}{}'.format(mount_root, str_all.find(dev) * 10 + num_all.find(num))
        while os.path.exists(mount_point) and len(os.listdir(mount_point)) > 0:
            mount_point = mount_point[:-1] + chr(ord(mount_point[len(mount_point) - 1]) + 1)
        return mount_point

    @staticmethod
    def _is_eligible_to_mount(disk_label):
        """Check if disk is eligible to mount, if not, fail silently"""
        valid_mount_cols = 4
        mount_info = utils.check_output('lsblk | grep {}'.format(disk_label.split('/')[-1]))
        return (len(mount_info) == 1 and len(mount_info[0].split()) >= valid_mount_cols and (
            mount_info[0].split()[valid_mount_cols - 1].endswith(b'T') or
            mount_info[0].split()[valid_mount_cols - 1].endswith(b'G')))


