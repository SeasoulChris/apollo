#!/usr/bin/env python
"""File related utils."""

import errno
import os

import colored_glog as glog


def makedirs(dir_path):
    """Make directories recursively."""
    if os.path.exists(dir_path):
        return dir_path
    try:
        os.makedirs(dir_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            glog.error('Failed to makedir ' + dir_path)
            raise
    return dir_path


def touch(file_path):
    """Touch file."""
    makedirs(os.path.dirname(file_path))
    try:
        if not os.path.exists(file_path):
            glog.info('Touch file: {}'.format(file_path))
            os.mknod(file_path)
    except:
        glog.error('Failed to touch file ' + file_path)
        raise
    return file_path
