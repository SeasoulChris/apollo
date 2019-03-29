import os

import h5py
import numpy as np

import fueling.common.file_utils as file_utils

def write_h5(elem, folder_path, file_name):
    """write to h5 file, use feature key as file name"""
    file_utils.makedirs(folder_path)
    with h5py.File("{}/{}.hdf5".format(folder_path, file_name), "w") as out_file:
        i = 0
        for data_set in elem:
            name = "_segment_" + str(i).zfill(3)
            out_file.create_dataset(name, data=data_set, dtype="float32")
            i += 1

def write_h5_single_segment(data, file_dir, file_name):
    file_utils.makedirs(file_dir)
    with h5py.File("{}/{}.hdf5".format(file_dir, file_name), "w") as h5_file:
        h5_file.create_dataset("segment", data=data, dtype="float32")


def read_h5(file_path):
    """
    load h5 file to a numpy array
    """
    segment = None
    with h5py.File(file_path, 'r') as fin:
        for ds in fin.itervalues():
            if segment is None:
                segment = np.array(ds)
            else:
                segment = np.concatenate((segment, np.array(ds)), axis=0)
    return segment