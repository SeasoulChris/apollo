#!/usr/bin/env python
"""
common functions for feature extractin
"""

import glob

from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto.localization_pb2 import LocalizationEstimate


def folder_to_record(pathname):
    """ folder path to record path"""
    return glob.glob(pathname + "/*.record.*")


def process_msg(msg):
    """Parse message"""
    if msg.topic == "/apollo/canbus/chassis":
        msg_new = Chassis()
    else:
        msg_new = LocalizationEstimate()
    msg_new.ParseFromString(msg.message)
    return msg_new


def gen_key(elem):
    """ generate a key contians both folder path and time stamp """
    return ((elem[0], (int)(elem[1].header.timestamp_sec/60)), elem[1])


def process_seg(elem):
    """group Chassis and Localization msgs to list seperately"""
    chassis = []
    pose = []
    for each_elem in elem:
        if each_elem.header.module_name == "canbus":
            chassis.append(each_elem)
        else:
            pose.append(each_elem)
    return (chassis, pose)


def to_list(elem):
    """convert element to list"""
    return [elem]


def append(orig_elem, app_elem):
    """append another element to the revious element"""
    orig_elem.append((app_elem))
    return orig_elem


def extend(orig_elem, app_elem):
    """extend the original list"""
    orig_elem.extend(app_elem)
    return orig_elem
