#!/usr/bin/env python
import glob

import h5py
import numpy as np
import pyspark_utils.helper as spark_helper

from cyber_py.record import RecordReader
import fueling.common.record_utils as record_utils
from fueling.control.features.features import GetDatapoints
from modules.canbus.proto.chassis_pb2 import Chassis
from modules.localization.proto.localization_pb2 import LocalizationEstimate


MAX_PHASE_DELTA = 0.01
MIN_SEGMENT_LENGTH = 10

WantedChannels = ['/apollo/canbus/chassis',
                  '/apollo/localization/pose']

pathname = "/apollo/modules/data/fuel/fueling/control/records/"


def FoldToRecords(pathname):
    return glob.glob(pathname + "/*.record.*")


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
    """align chassis and pose data and build data segment"""
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
    """ write data segment to hdf5 file """
    time_stamp = str((int)(elem[0][1].header.timestamp_sec))
    outFile = h5py.File(
        "./training_dataset_{}.hdf5".format(time_stamp), "w")
    chassis = elem[0]
    pose = elem[1]
    i = 0
    res = []
    for mini_dataset in BuildTrainingDataset(chassis, pose):
        name = "_segment_" + str(i).zfill(3)
        outFile.create_dataset(name, data=mini_dataset, dtype="float32")
        i += 1
        res.append(mini_dataset)
    outFile.close()
    return len(res[0]), len(res[0][0])


if __name__ == '__main__':

    channels = (spark_helper.get_context('Test')
                # folder path
                .parallelize([pathname])
                # record path
                .flatMap(FoldToRecords)
                # read message
                .flatMap(record_utils.read_record(WantedChannels))
                # parse message
                .map(processMSG))

    pre_segment = (channels
                   # choose time as key, group msg into 1 sec
                   .keyBy(lambda x: (int)(x.header.timestamp_sec))
                   # combine chassis message and pose message with the same key
                   .combineByKey(to_list, append, extend))

    data = (pre_segment
            # msg list
            .mapValues(processSeg)
            # align msg, generate data segment, write to hdf5 file.
            .mapValues(Run))
    print data.count()
