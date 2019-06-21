#!/usr/bin/env python3

"""Execute the uploading with rsync way"""

import os

from executor import Executor
import global_settings as settings
import utils as utils

class SerializeJobTaskExecutor(Executor):
    """Execute syncing."""
    def __init__(self, src_dst_map):
        Executor.__init__(self, src_dst_map)
        self._bos_root = 'bos:/apollo-platform'
        self._conf = utils.load_yaml_settings('conf/uploader_conf.yaml')

    def before_sync_all(self, task_id, logger):
        """Run any possible commands for preparations before sync everything"""
        # NoOp for SerializeJobTaskExecutor
        pass

    def before_sync_one(self, src, dst, logger):
        """Run any possible commands for preparations before sync each single target"""
        # NoOp for SerializeJobTaskExecutor
        pass

    def after_sync_one(self, src, dst, logger):
        """Run any possible commands for closure after sync each single target"""
        # NoOp for SerializeJobTaskExecutor
        pass

    def after_sync_all(self, task_id, logger):
        """Run any possible commands for closure after sync everything"""
        task_dir = settings.get_param(task_id).Root
        # Copy the record file to dst
        src_streaming_records_path = os.path.join(self._conf['src_streaming_path'], 'records')
        dst_streaming_records_path = os.path.join(self._conf['dst_streaming_path'], 'records')
        logger.log('copying record file')
        self.do_sync(os.path.join(src_streaming_records_path, os.path.basename(task_dir)),
                     os.path.join(dst_streaming_records_path, os.path.basename(task_dir)), logger)
        # Remove src task from data path
        if os.path.exists(task_dir) and str(task_dir).startswith(self._conf['src_streaming_path']):
            removing_cmd = 'rm -fr {}'.format(task_dir)
            logger.log('after work removing: {}'.format(removing_cmd))
            os.system(removing_cmd)

    def do_sync(self, src, dst, logger):
        """Actually sync"""
        remote_sync_cmd = 'ssh {}@{} bash {} {} {}'.format(
            self._conf['remote_user'], self._conf['remote_ip'], self._conf['src_sync_script_path'],
            src, os.path.join(self._bos_root, dst))
        logger.log('calling remote command: {}'.format(remote_sync_cmd))
        os.system(remote_sync_cmd)
