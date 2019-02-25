import h5py


def write_h5(elem, folder_path, key):
    """write to h5 file, use feature key as file name"""
    out_file = h5py.File(
        "{}/dataset_{}.hdf5".format(folder_path, key), "w")
    i = 0
    for data_set in elem:
        name = "_segment_" + str(i).zfill(3)
        out_file.create_dataset(name, data=data_set, dtype="float32")
        i += 1
    out_file.close()
