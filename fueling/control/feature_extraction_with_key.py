#!/usr/bin/env python
"""
This is a module to extraction features from records
with folder path as part of the key
"""

import glob

import h5py
import numpy as np

import fueling.common.record_utils as record_utils
import fueling.common.spark_utils as spark_utils
from fueling.control.features.features import GetDatapoints
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto.localization_pb2 import LocalizationEstimate

MAX_PHASE_DELTA = 0.01
MIN_SEGMENT_LENGTH = 10


PATH = ["/apollo/modules/data/fuel/fueling/control/records/Transient_1",
        "/apollo/modules/data/fuel/fueling/control/records/Transient_2"]

WANTED_CHS = ['/apollo/canbus/chassis',
              '/apollo/localization/pose']


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


def gen_folder_time_key(elem):
    """ generate a key contians both folder path and time stamp """
    return ((elem[0], (int)(elem[1].header.timestamp_sec/60)), elem[1])


def to_list(elem):
    """convert element to list"""
    return [(elem)]


def append(orig_elem, app_elem):
    """append another element to the revious element"""
    orig_elem.append((app_elem))
    return orig_elem


def extend(orig_elem, app_elem):
    """extend the original list"""
    orig_elem.extend(app_elem)
    return orig_elem


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


def build_training_dataset(chassis, pose):
    """align chassis and pose data and build data segment"""
    chassis.sort(key=lambda x: x.header.timestamp_sec)
    pose.sort(key=lambda x: x.header.timestamp_sec)
    # In the record, control and chassis always have same number of frames
    times_pose = np.array([x.header.timestamp_sec for x in pose])
    times_cs = np.array([x.header.timestamp_sec for x in chassis])

    index = [0, 0]

    def align():
        """align up chassis and pose data w.r.t time """
        while index[0] < len(times_cs) and index[1] < len(times_pose) \
                and abs(times_cs[index[0]] - times_pose[index[1]]) > MAX_PHASE_DELTA:
            while index[0] < len(times_cs) and index[1] < len(times_pose) \
                    and times_cs[index[0]] < times_pose[index[1]] - MAX_PHASE_DELTA:
                index[0] += 1
            while index[0] < len(times_cs) and index[1] < len(times_pose) \
                    and times_pose[index[1]] < times_cs[index[0]] - MAX_PHASE_DELTA:
                index[1] += 1

    align()

    while index[0] < len(times_cs)-1 and index[1] < len(times_pose)-1:
        limit = min(len(times_cs)-index[0], len(times_pose)-index[1])

        for seg_len in range(1, limit):
            delta = abs(times_cs[index[0]+seg_len]
                        - times_pose[index[1]+seg_len])
            if delta > MAX_PHASE_DELTA or seg_len == limit-1:
                if seg_len >= MIN_SEGMENT_LENGTH or seg_len == limit - 1:
                    yield GetDatapoints(pose[index[1]:index[1]+seg_len],
                                        chassis[index[0]:index[0]+seg_len])
                    index[0] += seg_len
                    index[1] += seg_len
                    align()
                    break


def run(elem):
    """ write data segment to hdf5 file """
    time_stamp = str(elem[0][1])
    folder_path = str(elem[0][0])
    out_file = h5py.File(
        "{}/training_dataset_{}.hdf5".format(folder_path, time_stamp), "w")
    chassis = elem[1][0]
    pose = elem[1][1]
    i = 0
    for mini_dataset in build_training_dataset(chassis, pose):
        name = "_segment_" + str(i).zfill(3)
        out_file.create_dataset(name, data=mini_dataset, dtype="float32")
        i += 1
    out_file.close()
    return i


CHANNELS = (spark_utils.get_context('Test')
            # folder path
            .parallelize(PATH)
            # add foler path as key
            .keyBy(lambda x: x)
            # record path
            .flatMapValues(folder_to_record)
            # read message
            .flatMapValues(record_utils.read_record(WANTED_CHS))
            # parse message
            .mapValues(process_msg))

PRE_SEGMENT = (CHANNELS
               # choose time as key, group msg into 1 sec
               .map(gen_folder_time_key)
               # combine chassis message and pose message with the same key
               .combineByKey(to_list, append, extend))


DATA = (PRE_SEGMENT
        # msg list(path_key,(chassis,pose))
        .mapValues(process_seg)
        # align msg, generate data segment, write to hdf5 file.
        .map(run))
print DATA.first()
