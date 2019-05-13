#!/usr/bin/env python
import os

import colored_glog as glog


def hdf52txt(hdf5_file, txt_file):
    data = generate_segment_from_list(hdf5_files)
    np.savetxt(txt_file, input_data, delimiter=' ')


def echo_lincoln(input_file, output_file):
    """Call echo_lincoln C++ code."""
    command = (
        'bash '
        '/apollo/modules/data/fuel/fueling/control/scripts/echo_lincoln_offline.sh '
        '"{}" "{}"'.format(input_file, output_file))
    if os.system(command) == 0:
        glog.info('Generated results')
        return 1
    else:
        glog.error('Failed to generate results')
    return 0


# # usage
# if __name__ == '__main__':
#     input_file = "/apollo/data/hdf52txt/2_3.txt"
#     output_file = "/apollo/data/hdf52txt/2_3_result.txt"
#     echo_lincoln(input_file, output_file)
