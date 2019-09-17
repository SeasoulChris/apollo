#!/usr/bin/env python
"""File related utils."""

import errno
import os

import fueling.common.logging as logging


def makedirs(dir_path):
    """Make directories recursively."""
    if os.path.exists(dir_path):
        return dir_path
    try:
        os.makedirs(dir_path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            logging.error('Failed to makedir ' + dir_path)
            raise
    return dir_path


def touch(file_path):
    """Touch file."""
    makedirs(os.path.dirname(file_path))
    try:
        if not os.path.exists(file_path):
            logging.info('Touch file: {}'.format(file_path))
            os.mknod(file_path)
    except:
        logging.error('Failed to touch file ' + file_path)
        raise
    return file_path


def list_files(dir_path):
    """List all sub-files in given dir_path."""
    return [os.path.join(root, f) for root, _, files in os.walk(dir_path) for f in files]
