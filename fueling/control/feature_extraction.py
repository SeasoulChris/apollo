#!/usr/bin/env python
import datetime
import operator
import pprint
import fueling.common.record_utils as record_utils
import fueling.common.spark_utils as spark_utils

from fueling.control.features.features import GetDatapoints

import glob
import os
import sys
import glog

import h5py
from cyber_py import cyber
from cyber_py import record
from google.protobuf.descriptor_pb2 import FileDescriptorProto
import numpy as np
import argparse
import time
from cyber_py.record import RecordReader

from modules.canbus.proto.chassis_pb2 import Chassis
from modules.control.proto.control_cmd_pb2 import ControlCommand
from modules.localization.proto.localization_pb2 import LocalizationEstimate

MAX_PHASE_DELTA = 0.01
MIN_SEGMENT_LENGTH = 10

WantedChannels = ['/apollo/canbus/chassis',
                  '/apollo/localization/pose']

path = os.path.join("/apollo/modules/data/fuel/fueling/control/records/",
                    "20190201110438.record.00130")

pathname = "/apollo/modules/data/fuel/fueling/control/records/"


def FoldToRecords(pathname):
    return glob.glob(pathname + "/*.record.*")


def ProcessChassisMsg(msg):
    msg_new_cs = Chassis()
    msg_new_cs.ParseFromString(msg[1])
    return msg_new_cs


def ProcessPoseMsg(msg):
    msg_new_pose = LocalizationEstimate()
    msg_new_pose.ParseFromString(msg[1])
    return msg_new_pose


def to_list(a):
    return [(a)]


def append(a, b):
    a.append((b))
    return a


def extend(a, b):
    a.extend(b)
    return a


def processMSG(msg):
    if msg.topic == "/apollo/canbus/chassis":
        msg_new = Chassis()
    else:
        msg_new = LocalizationEstimate()
    msg_new.ParseFromString(msg.message)
    return msg_new


def processSeg(elem):
    chassis = []
    pose = []
    for x in elem:
        if x.header.module_name == "canbus":
            chassis.append(x)
        else:
            pose.append(x)
    return (chassis, pose)


def BuildTrainingDataset(chassis, pose):
    chassis.sort(key=lambda x: x.header.timestamp_sec)
    pose.sort(key=lambda x: x.header.timestamp_sec)
    # In the record, control and chassis always have same number of frames
    times_pose = np.array([x.header.timestamp_sec for x in pose])
    times_cs = np.array([x.header.timestamp_sec for x in chassis])

    index = [0, 0]

    def Align():
        while index[0] < len(times_cs) and index[1] < len(times_pose) \
                and abs(times_cs[index[0]] - times_pose[index[1]]) > MAX_PHASE_DELTA:
            while index[0] < len(times_cs) and index[1] < len(times_pose) \
                    and times_cs[index[0]] < times_pose[index[1]] - MAX_PHASE_DELTA:
                index[0] += 1
            while index[0] < len(times_cs) and index[1] < len(times_pose) \
                    and times_pose[index[1]] < times_cs[index[0]] - MAX_PHASE_DELTA:
                index[1] += 1

    Align()

    while index[0] < len(times_cs)-1 and index[1] < len(times_pose)-1:
        limit = min(len(times_cs)-index[0], len(times_pose)-index[1])

        for d in range(1, limit):
            delta = abs(times_cs[index[0]+d] - times_pose[index[1]+d])
            if delta > MAX_PHASE_DELTA or d == limit-1:
                if d >= MIN_SEGMENT_LENGTH or d == limit - 1:
                    # TODO: check record data
                    yield GetDatapoints(pose[index[1]:index[1]+d], chassis[index[0]:index[0]+d])
                    index[0] += d
                    index[1] += d
                    Align()
                    break


def Run(elem):
    time_stamp = str((int)(elem[0][1].header.timestamp_sec))
    outFile = h5py.File(
        "./training_dataset_{}.hdf5".format(time_stamp), "w")
    chassis = elem[0]
    pose = elem[1]
    i = 0
    for mini_dataset in BuildTrainingDataset(chassis, pose):
        name = "_segment_" + str(i).zfill(3)
        outFile.create_dataset(name, data=mini_dataset, dtype="float32")
        i += 1
    return i


if __name__ == '__main__':

    channels = (spark_utils.GetContext('Test')
                .parallelize([pathname])  # folder
                .flatMap(FoldToRecords)  # record
                .flatMap(record_utils.ReadRecord(WantedChannels))
                .map(processMSG))

    pre_segment = (channels
                   .keyBy(lambda x: (int)(x.header.timestamp_sec))
                   .combineByKey(to_list, append, extend))

    data = (pre_segment
            .mapValues(processSeg)
            .mapValues(Run))
    print data.count()
