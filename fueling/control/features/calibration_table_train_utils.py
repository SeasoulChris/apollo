import h5py
import numpy as np
import random
import glog
import glob

from fueling.control.features.filters import Filters
import fueling.common.h5_utils as h5_utils
import fueling.common.file_utils as file_utils
from neural_network_tf import NeuralNetworkTF


def choose_data_file(elem, vehicle_type, brake_or_throttle, train_or_test):
    dir = elem[0]
    hdf5_file = glob.glob(
        '{}/{}_{}_{}_*.hdf5'.format(dir, vehicle_type, brake_or_throttle, train_or_test))
    return (elem[0], hdf5_file)


def generate_segments(h5s):
    segments = []
    for h5 in h5s:
        print('Loading {}'.format(h5))
        with h5py.File(h5, 'r+') as f:
            names = [n for n in f.keys()]
            print('f.keys', f.keys())
            if len(names) < 1:
                continue
            for i in range(len(names)):
                ds = np.array(f[names[i]])
                segments.append(ds)
    # shuffle(segments)
    print('Segments count: ', len(segments))
    return segments


def generate_data(segments):
    """ combine data from each segments """
    total_len = 0
    for i in range(len(segments)):
        total_len += segments[i].shape[0]
    print("total_len = ", total_len)
    dim_input = 2
    dim_output = 1
    X = np.zeros([total_len, dim_input])
    Y = np.zeros([total_len, dim_output])
    i = 0
    for j in range(len(segments)):
        segment = segments[j]
        for k in range(segment.shape[0]):
            if k > 0:
                X[i, 0:2] = segment[k, 0:2]
                Y[i, 0] = segment[k, 2]
                i += 1
    return X, Y


def train_model(elem, layer, train_alpha):
    """
    train model
    """
    X_train = elem[0][0]
    Y_train = elem[0][1]
    X_test = elem[1][0]
    Y_test = elem[1][1]

    model = NeuralNetworkTF(layer)
    params, train_cost, test_cost = model.train(X_train, Y_train,
                                                X_test, Y_test,
                                                alpha=train_alpha,
                                                print_loss=True)
    glog.info(" model train cost: %d" % train_cost)
    glog.info(" model test cost: %d " % test_cost)
