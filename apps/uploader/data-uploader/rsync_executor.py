#!/usr/bin/env python3

"""Execute the uploading with rsync way"""

import os

from executor import Executor
import utils as utils


class RsyncExecutor(Executor):
    """Execute syncing."""

    def __init__(self, src_dst_map):
        Executor.__init__(self, src_dst_map)
        self._bos_root = '/mnt/bos'

    def before_sync_all(self, task_id, logger):
        """Run any possible commands for preparations before sync everything"""
        # NoOp for RsyncExecutor
        pass

    def before_sync_one(self, src, dst, logger):
        """Run any possible commands for preparations before sync each single target"""
        logger.log('making dir for dst: {}'.format(dst))
        utils.makedirs(dst)

    def after_sync_one(self, src, dst, logger):
        """Run any possible commands for closure after sync each single target"""
        # NoOp for RsyncExecutor
        pass

    def do_sync(self, src, dst, logger):
        """Actually sync"""
        rsync_opts = '-ah --append-verify --progress --no-perms --omit-dir-times --delete-after'
        sync_cmd = 'rsync {} {}/ {}/'.format(rsync_opts, src, os.path.join(self._bos_root, dst))
        logger.log('rsyncing: {}'.format(sync_cmd))
        os.system(sync_cmd)
