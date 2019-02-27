import os

import h5py


def write_h5(elem, folder_path, file_name):
    """write to h5 file, use feature key as file name"""
    if (os.path.exists(folder_path) == False):
        os.mkdir(folder_path)
    out_file = h5py.File(
        "{}/{}.hdf5".format(folder_path, file_name), "w")
    i = 0
    for data_set in elem:
        name = "_segment_" + str(i).zfill(3)
        out_file.create_dataset(name, data=data_set, dtype="float32")
        i += 1
    out_file.close()
