#!/usr/bin/env python
import os
import glob
import time

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
import h5py
import matplotlib.pyplot as plt
import numpy as np

# speed_max = 25; speed_step = 5; min_speed_key = 1
# (0~5) (5~10) (10~15) (15~20) (20~25) 5
# steering_max = 100; steering_min = -100; step = 20; min_key = 0
# (-100~-80) (-80~-60) (-60~-40) (-40~-20) (-20~0)
# (0 ~ 20) (20 ~ 40) (40 ~ 60)
# (0~20) (20~40) () 10
# throttle_max = 45; step 5; min_key = 0
# (dead_zone - dead_zone+5) (dead_zone+5 - dead_zone+10) (dead_zone+10 - dead_zone+15) ... 6
# brake ... 5


def all_key(speed_num, steering_num, throttle_num, brake_num):
    """ generate key """
    total_key = speed_num * steering_num * (throttle_num + brake_num)
    # key_list = np.zeros((total_key, 1))
    key_list = []
    print(total_key)
    # print(key_list.shape)
    count = 0
    for i_speed in range(1, speed_num):
        for i_steering in range(steering_num):
            for i_throttle in range(throttle_num):
                # print(count)
                key_list.append(str(i_speed * 1000 + i_steering * 100 + i_throttle * 10))
                count += 1
            for i_brake in range(brake_num):
                key_list.append(str(i_speed * 1000 + i_steering * 100 + i_brake))
                count += 1
    return key_list


def read_hdf5(folder):
    """
    load h5 file to a numpy array
    """
    segment = None
    # for filename in glob.iglob(os.path.join(folder, '**/*.hdf5'), recursive=True):
    for filename in glob.glob('./*.hdf5'):
        with h5py.File(filename, 'r') as fin:
            for value in fin.value():
                if segment is None:
                    segment = np.array(value)
                else:
                    segment = np.concatenate((segment, np.array(value)), axis=0)
    return segment


def gen_feature(data):
    return data[:, (14, 15, 16, 17)]


def plot_feature_hist(fearure):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pdf_file = ('./Dataset_Distribution_%s.pdf' % timestr)
    with PdfPages(pdf_file) as pdf:
        for j in range(DIM_INPUT):
            plt.figure(figsize=(4, 3))
            plt.hist(fearure[:, j], bins='auto')
            plt.title("Histogram of the " + list(input_index)[j])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


DIM_INPUT = 4

input_index = {
    "speed": 0,  # chassis.speed_mps
    "throttle": 1,  # chassis.throttle_percentage/100.0
    "brake": 2,  # chassis.brake_percentage/100.0
    "steering": 3  # chassis.steering_percentage/100.
}

if __name__ == '__main__':
    # speed_num = 5
    # steering_num = 10
    # throttle_num = 6
    # brake_num = 5
    # key_list = all_key(speed_num, steering_num, throttle_num, brake_num)
    # print(key_list)
    data_set = read_hdf5("./result_hdf5/uniform_distribute_data/")
    feature = gen_feature(data_set)
    plot_feature_hist(feature)
    print(feature.shape)
