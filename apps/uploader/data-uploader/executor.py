#!/usr/bin/env python3

"""Execute the uploading"""

from collections import namedtuple

import global_settings as settings
import os
import time
import utils as utils


class Executor(object):
    """Execute syncing."""
    # Static namedtuple defination
    Statistic = namedtuple('Statistic',
                           ['Source', 'Files', 'Size', 'Destination', 'Estimate', 'Time', 'Speed'])

    def __init__(self, src_dst_map):
        # src_dst_map: src->dst, src is tuple (src_path, file_num, size)
        # It's part of this one (src, files_num, size, dst, estimate_speed, time, actual_speed)
        Executor.Statistic.__new__.__defaults__ = (None,) * len(Executor.Statistic._fields)
        self._statistics = []
        self._bos_root = ''
        for src, dst in src_dst_map:
            self._statistics.append(Executor.Statistic(
                Source=src[0], Files=src[1], Size=src[2],
                Destination=dst, Estimate=utils.get_spent_time(src[2], 40.0)
            ))

    def get_stastic_before_executing(self):
        """Return statistics before executing"""
        ProcessingStat = namedtuple('ProcessingStat',
                                    ['Source', 'Files', 'Size', 'Destination', 'Estimate'])
        content = []
        total_size = 0.0
        total_files = 0
        total_time = 0
        for statistic in self._statistics:
            content.append(ProcessingStat(
                Source=statistic.Source,
                Files=statistic.Files,
                Size=statistic.Size,
                Destination=statistic.Destination,
                Estimate=statistic.Estimate
            ))
            total_size += utils.get_size_in_mb(statistic.Size)
            total_files += int(statistic.Files)
            total_time += utils.get_time_seconds(statistic.Estimate)
        content.append(ProcessingStat(
            Source='Total',
            Files=str(total_files),
            Size='{:.1f}G'.format(total_size / 1024),
            Destination='',
            Estimate=utils.get_readable_time(total_time)
        ))
        return content

    def get_stastic_after_executing(self):
        """Return statistics after executing"""
        CompleteStat = namedtuple('CompleteStat', ['Source', 'Destination', 'Speed', 'Time'])
        content = []
        total_size = 0.0
        total_time = 0
        for statistic in self._statistics:
            content.append(CompleteStat(
                Source=statistic.Source,
                Destination=statistic.Destination,
                Speed=statistic.Speed,
                Time=statistic.Time
            ))
            total_size += utils.get_size_in_mb(statistic.Size)
            total_time += utils.get_time_seconds(statistic.Time)
        content.append(CompleteStat(
            Source='Total',
            Destination='',
            Speed='{:.2f} MB/s'.format(total_size / (total_time if total_time != 0 else 1)),
            Time=utils.get_readable_time(total_time)
        ))
        return content

    def execute(self, task_id, logger):
        """Just do it"""
        if not self._statistics:
            error_msg = 'Nothing to Copy, Please check if folders are in correct location'
            logger.log(error_msg)
            settings.set_param(
                task_id,
                settings.get_param(task_id)._replace(ErrorCode=settings.ErrorCode.NO_CONTENT,
                                                     ErrorMsg=error_msg))
            return
        self.before_sync_all(task_id, logger)
        for idx, statistic in enumerate(self._statistics):
            retries = 2
            time_start = time.time()
            while retries > 0:
                retries -= 1
                try:
                    self.before_sync_one(statistic.Source, statistic.Destination, logger)
                    self.do_sync(statistic.Source, statistic.Destination, logger)
                    self.after_sync_one(statistic.Source, statistic.Destination, logger)
                    time_spent = time.time() - time_start
                    self._statistics[idx] = statistic._replace(
                        Time=utils.get_readable_time(time_spent),
                        Speed=utils.get_speed(time_spent, statistic.Size))
                    break
                except Exception as ex:
                    if retries <= 0:
                        settings.set_param(
                            task_id,
                            settings.get_param(task_id)._replace(
                                ErrorCode=settings.ErrorCode.FAIL,
                                ErrorMsg='failed to copy data from {} to {} with reason {}'.format(
                                    statistic.Source, statistic.Destination, ex)))
                    return
        self.after_sync_all(task_id, logger)
        settings.set_param(
            task_id, settings.get_param(task_id)._replace(ErrorCode=settings.ErrorCode.SUCCESS))

    def before_sync_all(self, task_id, logger):
        """Run any possible commands for preparations before sync everything"""
        raise Exception('before_sync_all is Not Implemented for base class')

    def before_sync_one(self, src, dst, logger):
        """Run any possible commands for preparations before sync each single target"""
        raise Exception('before_sync_one is Not Implemented for base class')

    def after_sync_one(self, src, dst, logger):
        """Run any possible commands for closure after sync each single target"""
        raise Exception('after_sync_one is Not Implemented for base class')

    def after_sync_all(self, task_id, logger):
        """Run any possible commands for closure after sync everything"""
        # Notify Serialize_Records job
        conf = utils.load_yaml_settings('conf/uploader_conf.yaml')
        dst_root = '/mnt/bos'
        src_streaming_record_path = os.path.join(conf['src_streaming_path'], 'records')
        local_record_path = 'records_tmp'
        utils.makedirs(local_record_path)
        for statistic in self._statistics:
            local_record_file = os.path.join(local_record_path, os.path.basename(statistic.Source))
            records = sorted(list(os.listdir(statistic.Source)))
            if not all(x.find('.record.') > 0 for x in records):
                continue
            with open(local_record_file, 'w') as record_file:
                for record in records:
                    record_file.write(os.path.join(dst_root, statistic.Destination, record) + '\n')
            copy_cmd = 'cp {} {}/'.format(local_record_file, src_streaming_record_path)
            logger.log('copying record to kick off serialize job, cmd: {}'.format(copy_cmd))
            os.system(copy_cmd)
        # Move src folder to upper level
        task_setting = settings.get_param(task_id)
        work_dir = os.path.join(task_setting.Mount, task_setting.Root)
        move_to_dir = os.path.dirname(work_dir)
        for folder in os.listdir(work_dir):
            move_to_name = '{}-UPLOADED'.format(folder)
            moving_command = 'mv {} {}'.format(os.path.join(work_dir, folder),
                                               os.path.join(move_to_dir, move_to_name))
            logger.log('after work moving: {}'.format(moving_command))
            os.system(moving_command)

    def do_sync(self, src, dst, logger):
        """Actually sync"""
        raise Exception('do_sync is Not Implemented for base class')
