#!/usr/bin/env python

import argparse
from fnmatch import fnmatch
import os
from shutil import copyfile

from absl import flags
from tqdm import tqdm

from fueling.common import file_utils


def siren_data_clustering(data_dir):
    '''Cluster into siren voice and nonsiren voice, and renaming'''
    siren_dir = file_utils.makedirs(os.path.join(data_dir, 'siren'))
    nonsiren_dir = file_utils.makedirs(os.path.join(data_dir, 'nonsiren'))

    files = file_utils.list_files(data_dir)
    siren_count = 0
    nonsiren_count = 0
    for file in tqdm(files):
        file_name = os.path.basename(os.path.normpath(file))
        if file.find('.wav') == -1:
            continue
        if fnmatch(file, '*-8-*-*.wav'):
            copyfile(file, os.path.join(siren_dir, 'Emergency_' + file_name))
            siren_count += 1
        else:
            copyfile(file, os.path.join(
                nonsiren_dir, 'nonEmergency_' + file_name))
            nonsiren_count += 1

    print('number of siren data is ' + str(siren_count))
    print('number of nonsiren data is ' + str(nonsiren_count))


if __name__ == "__main__":

    flags.DEFINE_string('data_dir', '/home/jinyun/Data/UrbanSound8K',
                        'data dir of UrbanSound8k data set.')

    def main(argv):

        flags_dict = flags.FLAGS.flag_values_dict()
        data_dir = flags_dict['data_dir']

        siren_data_clustering(data_dir)

    from absl import app
    app.run(main)
