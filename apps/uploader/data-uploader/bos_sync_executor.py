#!/usr/bin/env python3

"""Execute the uploading with bos sync way"""

import os

from executor import Executor
import utils as utils


class BosSyncExecutor(Executor):
    """Execute syncing."""

    def __init__(self, src_dst_map):
        Executor.__init__(self, src_dst_map)
        self._bos_root = 'bos:/apollo-platform'

    def before_sync_all(self, task_id, logger):
        """Run any possible commands for preparations before sync everything"""
        # NoOp for BosSyncExecutor
        pass

    def before_sync_one(self, src, dst, logger):
        """Run any possible commands for preparations before sync each single target"""
        logger.log('cleaning symbolic links for {}'.format(src))
        utils.clean_symbolic_links(src)

    def after_sync_one(self, src, dst, logger):
        """Run any possible commands for closure after sync each single target"""
        # NoOp for BosSyncExecutor
        pass

    def do_sync(self, src, dst, logger):
        """Actually sync"""
        sync_cmd = 'bce bos sync {} {} --quiet'.format(src, os.path.join(self._bos_root, dst))
        logger.log('bos syncing: {}'.format(sync_cmd))
        os.system(sync_cmd)
