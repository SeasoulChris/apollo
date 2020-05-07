#!/usr/bin/env python
"""H5 utils."""
import os

import h5py
import numpy as np


import fueling.common.file_utils as file_utils
import fueling.common.logging as logging


def write_h5(elem, file_dir, file_name):
    """
    Write to h5 file, one sample per dataset.
    The {file_dir/file_name} combination must be unique.
    """
    file_utils.makedirs(file_dir)
    with h5py.File("{}/{}.hdf5".format(file_dir, file_name), "w") as h5_file:
        for count, data_set in enumerate(elem):
            name = "_segment_" + str(count).zfill(3)
            h5_file.create_dataset(name, data=data_set, dtype="float64")


def write_h5_single_segment(data, file_dir, file_name):
    """
    Write to h5 file with all data in one dataset.
    The {file_dir/file_name} combination must be unique.
    """
    file_utils.makedirs(file_dir)
    with h5py.File("{}/{}.hdf5".format(file_dir, file_name), "w") as h5_file:
        h5_file.create_dataset("segment", data=data, dtype="float64")


def read_h5(hdf5_file):
    """
    Load h5 file to a numpy array.
    """
    segment = None
    with h5py.File(hdf5_file, 'r') as fin:
        for value in fin.values():
            if segment is None:
                segment = np.array(value)
            else:
                segment = np.concatenate((segment, np.array(value)), axis=0)
    return segment


def combine_h5_to_npy(hdf5_file_list, dst_dir):
    """
    combine a list of sorted h5 files to a numpy array.
    """
    segment = None
    for hdf5_file in hdf5_file_list:
        if segment is None:
            segment = read_h5(hdf5_file)
        else:
            segment = np.concatenate((segment, read_h5(hdf5_file)), axis=0)
    dst_file = os.path.join(dst_dir, 'combined.npy')
    logging.info(f'{len(hdf5_file_list)} data points are saved to file {dst_file}')
    np.save(dst_file, np.array(segment))
