#!/usr/bin/env python
""" utils for transferring the profiling output into the json format """

import json
import os

import numpy as np

import fueling.common.file_utils as file_utils
import fueling.common.proto_utils as proto_utils


def write_json(out_data, folder_path, file_name):
    """write to jason file, use feature key as file name"""
    if not os.path.isdir(folder_path):
        file_utils.makedirs(folder_path)
    with open("{}/{}.json".format(folder_path, file_name), "w") as out_file:
        out_file.write(json.dumps(out_data))


def get_pb_from_numpy_array(np_array, pb_value):
    """Get a proto from a given numpy array """
    """Here we assume the data struture of the numpy array exactly matches the proto """
    fields = pb_value.DESCRIPTOR.fields
    for field_num in range(0, len(fields)):
        field_value = getattr(pb_value, fields[field_num].name)
        del field_value[:]
        field_value.extend([float('%.6f' % x) for x in np_array[:, field_num].tolist()])
