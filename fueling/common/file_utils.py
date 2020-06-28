#!/usr/bin/env python
"""File related utils."""

import errno
import os
import time

import fueling.common.logging as logging


FUEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


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
    except BaseException:
        logging.error('Failed to touch file ' + file_path)
        raise
    return file_path


def list_files(dir_path):
    """List all sub-files in given dir_path."""
    return [os.path.join(root, f) for root, _, files in os.walk(dir_path) for f in files]


def list_files_with_suffix(dir_path, suffix):
    """List all sub-files with suffix in given dir_path."""
    return [os.path.join(root, f) for root, _, files
            in os.walk(dir_path) for f in files if f.endswith(suffix)]


def file_exists(filename):
    """Check if specified file is existing, with retry in case the mounted dir has network delay."""
    for t in range(5):
        if os.path.exists(filename):
            return True
        elif t == 4:
            return False
        else:
            sleep_time_in_min = t + 1
            logging.info(f"Retry checking {filename} in {sleep_time_in_min} min...")
            time.sleep(60 * sleep_time_in_min)


def fuel_path(path):
    """Get real path to data which is relative to Apollo Fuel root."""
    return os.path.join(FUEL_ROOT, path)


def apollo_path(path):
    """Get real path to data which is relative to Apollo root."""
    return os.path.join('/apollo', path)


def formatSize(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except BaseException:
        logging.error('Failed to get file format!')
        raise

    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%.2fGB" % (G)
        else:
            return "%.2fMB" % (M)
    else:
        return "%.2fKB" % (kb)


def getDirSize(path):
    sumsize = 0
    filelist = list_files(path)
    for file in filelist:
        size = os.path.getsize(file)
        sumsize += size
    return formatSize(sumsize)


def getInputDirDataSize(path):
    sumsize = 0
    filelist = list_files(path)
    for file in filelist:
        size = os.path.getsize(file)
        sumsize += size
    return int(sumsize)
