#!/usr/bin/env python3

"""Execute the uploading"""

import os

import colored_glog as glog

import utils as utils


class Logger(object):
    """Logging util. Each task has its own log file therefore logger."""

    def __init__(self, cur_folder, log_file_name):
        log_folder = os.path.join(cur_folder, 'log')
        utils.makedirs(log_folder)
        self._log_file_path = os.path.join(log_folder, log_file_name)

    def log(self, message):
        """Do the logging"""
        glog.info(message)
        with open(self._log_file_path, 'a') as log_file:
            log_file.write(message + ' \n')
