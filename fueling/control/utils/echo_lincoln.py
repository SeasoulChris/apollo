#!/usr/bin/env python
import os

import numpy as np
import colored_glog as glog

import fueling.control.dynamic_model.data_generator.feature_extraction as feature_extraction

# first line is initial velocity
# second line is initial throttle, brake, gear


def hdf52txt(hdf5_file, txt_file):
    data = feature_extraction.generate_segment_from_list(hdf5_file)
    data_points = np.size(data, 0)
    init_v = data[0, 14]
    # dimension check
    if np.size(data, 1) > 22:  # gear info is included
        input_data = data[:, [15, 16, 17, 22]]  # (throttle, brake, steering, gear)
        # scale (throttle, brake, steering) to 100%
        input_data[:, 0:3] = input_data[:, 0:3] * 100
        # scale steering angle to [-720, 720]
        input_data[:, 2] = input_data[:, 2] * 4.7
    else:
        # generate fake gear, which is forward by default
        gear_col = np.ones((data_points, 1))
        # scale (throttle, brake, steering) to 100%
        input_data = np.append(data[:, 15:18] * 100, gear_col, axis=1)
        # scale steering angle to [-720, 720]
        input_data[:, 2] = input_data[:, 2] * 4.7
    # np.savetxt(txt_file, input_data[1:10, :], delimiter=' ')  # set 1:10 for test
    np.savetxt(txt_file, input_data[:, :], delimiter=' ')
    # insert initial velocity as the first line
    with open(txt_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(str(init_v) + '\n' + content)


def echo_lincoln(input_file, output_file):
    """Call echo_lincoln C++ code."""
    command = (
        'bash /apollo/modules/data/fuel/fueling/control/scripts/echo_lincoln_offline.sh '
        '"{}" "{}"'.format(input_file, output_file))
    if os.system(command) == 0:
        glog.info('Generated results')
        return 1
    else:
        glog.error('Failed to generate results')
    return 0


def txt2numpy(txt_file):
    output_data = np.loadtxt(fname=txt_file)
    return output_data


def echo_lincoln_wrapper(hdf5_file):
    input_file = hdf5_file + '.txt'
    glog.info(input_file)
    output_file = hdf5_file + '_out.txt'
    glog.info(output_file)
    hdf52txt([hdf5_file], input_file)
    echo_lincoln(input_file, output_file)
    with open(output_file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(output_file, 'w') as fout:
        fout.writelines(data[1:])
    print(output_file)
    return txt2numpy(output_file)  # numpy data


# # demo
# if __name__ == '__main__':
#     FILE = '/apollo/data/hdf52txt/2_3.hdf5'
#     result = echo_lincoln_wrapper(FILE)
#     print(result)
