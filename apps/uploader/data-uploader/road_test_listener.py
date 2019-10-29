#!/usr/bin/env python3

"""Listen to the interested events and update status when those happen"""

import os
import re
import shutil

from listener import DiskManagement
from listener import Listener
from listener import Status
from sqliter import SqlLite3_DB

import global_settings as settings
import utils as utils


class RoadTestTaskListener(Listener):
    """RoadTest listener that listens to SqlLite DB"""

    def __init__(self):
        Listener.__init__(self)
        SqlLite3_DB.create_table()

    def get_available_task(self):
        """Check DB and returns the one with status INITIAL"""
        self._collect_initial_tasks_from_db()
        return super().get_available_task()

    def update_task_status(self, task, status):
        """Update status for particular task"""
        super().update_task_status(task, status)
        if status == Status.PROCESSING:
            SqlLite3_DB.update(task, status)
        elif status == Status.DONE:
            SqlLite3_DB.delete(task)
            DiskManagement.unmount(task)

    def collect(self, task_id, task, logger):
        """Collect the tasks with src, dst and related info"""
        # Now task here is a disk label, like /dev/sdb2
        collect_map = []
        mount_point = DiskManagement.mount(task_id, task, logger)
        if not mount_point:
            return collect_map
        disk_root = 'data/bag'
        settings.set_param(
            task_id,
            settings.get_param(task_id)._replace(Disk=task, Mount=mount_point, Root=disk_root))
        src_root = os.path.join(mount_point, disk_root)
        src_dirs = sorted(utils.get_all_directories(src_root, ['UPLOADED', '_s']), reverse=True)
        logger.log('all src dirs: {}'.format(src_dirs))
        src_dst_map = self._get_src_dst_map(src_dirs)
        for src in src_dst_map:
            collect_map.append(((src, len(os.listdir(src)), utils.get_size(src)),
                                src_dst_map[src]))
        logger.log('all jobs: {}'.format(collect_map))
        return collect_map

    def _collect_initial_tasks_from_db(self):
        """Search DB and update tasks if there are anyones with initial status"""
        for entry in SqlLite3_DB.search_tasks_by_status(Status.INITIAL):
            self._tasks[entry[1]] = Status.INITIAL

    def _get_src_dst_map(self, src_dirs):
        """Get corresponding dst path based on given src path"""
        record_pattern = r'^\d{14}.record.\d{5}$'
        bag_pattern = r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_\d+.bag$'
        bag_folder_pattern = r'^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$'
        other_pattern = r'[\S]*(\d{4})-(\d{2})-(\d{2})[\S]*'
        src_dst_map = {}
        for src in src_dirs:
            all_files = list(os.listdir(src))
            other_folder = '{}-other'.format(src)
            records = [f for f in all_files if re.match(record_pattern, f)]
            if len(records) > 0:
                if len(records) < len(all_files):
                    self._move_rests_to_other(src,
                                              [f for f in all_files if f not in records], other_folder)
                    src_dst_map[other_folder] = 'public-test/{}'.format(
                        self._get_other_dst(other_folder, other_pattern))
                src_dst_map[src] = 'public-test/{}'.format(self._get_record_dst(records[0]))
                continue
            bags = [f for f in all_files if re.match(bag_pattern, f)]
            if len(bags) > 0:
                if len(bags) < len(all_files):
                    self._move_rests_to_other(src,
                                              [f for f in all_files if f not in bags], other_folder)
                    src_dst_map[other_folder] = 'public-test/{}'.format(
                        self._get_other_dst(other_folder, other_pattern))
                src_dst_map[src] = 'stale-rosbags/{}'.format(
                    self._get_bag_dst(src, bag_folder_pattern, bags[0]))
                continue
            src_dst_map[src] = 'public-test/{}'.format(self._get_other_dst(src, other_pattern))
        return src_dst_map

    def _move_rests_to_other(self, src, rests, other):
        """Move rests files to other"""
        utils.makedirs(other)
        for rest in rests:
            shutil.move(os.path.join(src, rest), other)

    def _get_record_dst(self, first_file_in_src):
        """Get dst for records folder"""
        YYYY = first_file_in_src[:4]
        MM = first_file_in_src[4:6]
        DD = first_file_in_src[6:8]
        hh = first_file_in_src[8:10]
        mm = first_file_in_src[10:12]
        ss = first_file_in_src[12:14]
        return '{}/{}-{}-{}/{}-{}-{}-{}-{}-{}'.format(YYYY, YYYY, MM, DD, YYYY, MM, DD, hh, mm, ss)

    def _get_bag_dst(self, src, folder_pattern, first_file_in_src):
        """Get dst for bags folder"""
        # bag format: 2019-04-18-15-18-27_39.bag
        # path format: 2019-04-18-16-26-25
        match_target = first_file_in_src
        if re.match(folder_pattern, src):
            match_target = src
        YYYY = match_target[:4]
        MM = match_target[5:7]
        DD = match_target[8:10]
        hh = match_target[11:13]
        mm = match_target[14:16]
        ss = match_target[17:19]
        return '{}/{}-{}-{}/{}-{}-{}-{}-{}-{}'.format(YYYY, YYYY, MM, DD, YYYY, MM, DD, hh, mm, ss)

    def _get_other_dst(self, src, other_pattern):
        """Get dst for others folder"""
        se = re.search(other_pattern, src, re.M | re.I)
        YYYY = se.group(1)
        MM = se.group(2)
        DD = se.group(3)
        return '{}/{}-{}-{}/OTHER/{}'.format(YYYY, YYYY, MM, DD, os.path.basename(src))
