#!/usr/bin/env python3

"""Global settings that can be shared and used accross files"""

from collections import namedtuple
from enum import Enum


class ErrorCode(Enum):
    """ErrorCode enum."""
    SUCCESS = 0
    FAIL = 1
    NOT_ELIGIBLE = 2
    NO_CONTENT = 3


def init():
    """Holds some global parameters shared across files"""
    # task_id -> (error_code, error_message, disk_lable, mount_point, disk_root)
    global Param
    global task_params_map
    Param = namedtuple('Param', ['ErrorCode', 'ErrorMsg', 'Disk', 'Mount', 'Root'])
    Param.__new__.__defaults__ = (None,) * len(Param._fields)
    task_params_map = {}


def get_param(task_id):
    """Returns the corresponding param for given task_id if it exists, otherwise create one"""
    if task_id not in task_params_map:
        task_params_map[task_id] = Param()
    return task_params_map[task_id]


def set_param(task_id, param):
    """Set param for given task_id"""
    task_params_map.update({task_id: param})
