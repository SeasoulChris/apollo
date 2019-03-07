import h5py
import numpy as np
import random
import glog

from fueling.control.features.filters import Filters
import fueling.common.h5_utils as h5_utils


# calibration table constant
steer_condition = 1.0  # 1.0 for this case steering angle is large
curvature_condition = 0.01
speed_min_condition = 1.5
speed_max_condition = 17.0
acc_min_condition = -5.0
acc_max_condition = 5.0
train_percetage = 0.9

# Mkz7
segment_brake_list = np.linspace(-35.0, -21.0, num=11).tolist()
segment_throttle_list = np.linspace(18.0, 60.0, num=11).tolist()

# # transient
# segment_brake_list = np.linspace(-30.0, -7.0, num=11).tolist()
# segment_throttle_list = np.linspace(5.0, 30.0, num=11).tolist()

segment_speed_list = np.linspace(0.0, 18.0, num=11).tolist()
segment_cmd_list = segment_brake_list + segment_throttle_list
segment_store_num = 12

wanted_driving_mode = "COMPLETE_MANUAL"


def decide_cmd(chassis_throttle_val, chassis_brake_val):
    feature_cmd = 0.0
    if chassis_throttle_val > abs(segment_throttle_list[0]):
        feature_cmd = chassis_throttle_val
    elif chassis_brake_val > abs(segment_brake_list[-1]):
        feature_cmd = -chassis_brake_val
    else:
        feature_cmd = 0.0
    return feature_cmd


def feature_generate(elem):
    """ extract data from segment """
    res = np.zeros([len(elem), 5])
    for i in range(len(elem)):
        chassis = elem[i][0]
        pose = elem[i][1].pose
        feature_cmd = decide_cmd(
            chassis.throttle_percentage, chassis.brake_percentage)
        driving_mode = (chassis.driving_mode == wanted_driving_mode)
        res[i] = np.array([
            pose.linear_velocity.x,  # 0: speed
            pose.linear_acceleration.x,  # 1: acc
            feature_cmd,                # 2: cmd
            chassis.steering_percentage,
            driving_mode
        ])
    return res


def feature_filter(elem, filter_window=20):
    """
    filter feature with mean filter
    feature is one colm of elem
    """
    # elem is a matrix:
    # row: different type of features,
    # col: different value of a feature
    num_row = elem.shape[0]  # data points in a feature
    for i in range(3):
        # feature_num: numbers of data in one feature
        feature = [0 for k in range(num_row)]
        f = Filters(filter_window)
        for j in range(num_row):
            value = elem[j][i]
            feature[j] = f.mean_filter(value)
        elem[:, i] = feature
    return elem


def satisfy_brake_condition(elem, index):
    """
    whether satisfy brake condition
    """
    condition = abs(elem[index][3]) < steer_condition and \
        elem[index][0] > speed_min_condition and \
        elem[index][0] < speed_max_condition and \
        elem[index][2] > segment_brake_list[0] and \
        elem[index][2] < segment_brake_list[-1] and \
        elem[index][1] < 0.0 and \
        elem[index][1] > acc_min_condition and \
        int(elem[index][4]) == 0
    return condition


def satisfy_throttle_condition(elem, index):
    """
    whether satisfy throttle condition
    """
    condition = abs(elem[index][3]) < steer_condition and \
        elem[index][0] > speed_min_condition and \
        elem[index][0] < speed_max_condition and \
        elem[index][2] > segment_throttle_list[0] and \
        elem[index][2] < segment_throttle_list[-1] and \
        elem[index][1] < 0.0 and \
        elem[index][1] > acc_min_condition and \
        int(elem[index][4]) == 0
    return condition


def feature_cut(elem):
    """
    get desired feature interval
    """
    id_elem = 0
    num_row = elem.shape[0]
    # find satisfied data
    for i in range(num_row):
        if satisfy_throttle_condition(elem, i) or satisfy_brake_condition(elem, i):
            elem[id_elem][0] = elem[i][0]
            elem[id_elem][1] = elem[i][1]
            elem[id_elem][2] = elem[i][2]
            elem[id_elem][3] = elem[i][3]  # add steering angle as reference
            id_elem += 1

    return elem[0:id_elem, 0:4]


def feature_distribute(elem):
    """
    distribute feature into each grid
    """
    grid_dict = {}
    for segment_cmd in segment_cmd_list:
        grid_dict[segment_cmd] = {}
        for segment_speed in segment_speed_list:
            grid_dict[segment_cmd][segment_speed] = []

    # stratified storing data
    feature_num = elem.shape[0]  # number of rows
    for feature_index in range(feature_num):
        cmd = elem[feature_index][2]  # cmd --- 2
        speed = elem[feature_index][0]  # speed --- 0
        for cmd_index in range(len(segment_cmd_list) - 1):
            curr_segment_cmd = segment_cmd_list[cmd_index]
            next_segment_cmd = segment_cmd_list[cmd_index + 1]
            if (cmd > curr_segment_cmd and cmd < next_segment_cmd):
                for speed_index in range(len(segment_speed_list) - 1):
                    curr_segment_speed = segment_speed_list[speed_index]
                    next_segment_speed = segment_speed_list[speed_index + 1]
                    if (speed > curr_segment_speed and speed < next_segment_speed):
                        grid_dict[curr_segment_cmd][curr_segment_speed].append(
                            feature_index)
                        break
                break

    # delete data which exceeds average value too much
    for segment_cmd in segment_cmd_list:
        for segment_speed in segment_speed_list:
            feature_index_list = grid_dict[segment_cmd][segment_speed]
            if len(feature_index_list) == 0:
                continue
            acc_list = []
            for feature_index in feature_index_list:
                acc_list.append(elem[feature_index][1])
            acc_mean = np.mean(acc_list)
            acc_std = np.std(acc_list)
            for index, feature_index in enumerate(feature_index_list):
                if abs(elem[feature_index][1] - acc_mean) > acc_std:
                    grid_dict[segment_cmd][segment_speed].pop(index)

    # random sampling data
    for segment_cmd in segment_cmd_list:
        for segment_speed in segment_speed_list:
            feature_index_list = grid_dict[segment_cmd][segment_speed]
            store_num = min(len(feature_index_list), segment_store_num)
            feature_index_list = random.sample(feature_index_list, store_num)
            grid_dict[segment_cmd][segment_speed] = feature_index_list

    return (grid_dict, elem)


def feature_store(elem):
    """
    store feature into segment_feature container
    """
    grid_dict = elem[0]
    feature = elem[1]
    segment_feature = np.zeros([len(feature), 3])
    counter = 0
    for segment_cmd in segment_cmd_list:
        for segment_speed in segment_speed_list:
            for feature_index in grid_dict[segment_cmd][segment_speed]:
                # row: feature_index; col: 0:3
                segment_feature[counter] = feature[feature_index, 0:3]
                counter += 1
    return segment_feature[0:counter, :]


def write_h5_whole(elem):
    """write to h5 file, use feature key as file name"""
    key = str(elem[0][1])
    folder_path = str(elem[0][0])
    out_file = h5py.File(
        "{}/training_dataset_{}.hdf5".format(folder_path, key), "w")
    out_file.create_dataset("segment", data=elem[1], dtype="float32")
    out_file.close()
    return elem[0]


def write_h5_train_test(elem, origin_prefix, target_prefix, vehicle_type):
    """write to h5 file"""
    key = str(elem[0][1])
    folder_path = str(elem[0][0])
    feature = elem[1]
    feature_num = elem[1].shape[0]
    throttle_train_feature_num, throttle_test_feature_num = 0, 0
    brake_train_feature_num, brake_test_feature_num = 0, 0

    throttle_train = np.zeros(feature.shape)
    throttle_test = np.zeros(feature.shape)

    brake_train = np.zeros(feature.shape)
    brake_test = np.zeros(feature.shape)

    for i in range(feature_num):
        if feature[i][2] > 0.0:
            if random.random() < train_percetage:
                throttle_train[throttle_train_feature_num] = feature[i]
                throttle_train_feature_num += 1
            else:
                throttle_test[throttle_test_feature_num] = feature[i]
                throttle_test_feature_num += 1
        elif feature[i][2] < 0.0:
            if random.random() < train_percetage:
                brake_train[brake_train_feature_num] = feature[i]
                brake_train_feature_num += 1
            else:
                brake_test[brake_test_feature_num] = feature[i]
                brake_test_feature_num += 1

    # throttle train file
    glog.info('throttle file size: %d' % throttle_train.shape[0])
    glog.info('throttle train file size: %d' % throttle_train_feature_num)
    folder_path = folder_path = folder_path.replace(
        origin_prefix, target_prefix, 1)
    throttle_train_file = h5py.File(
        "{}/{}_throttle_train_{}.hdf5".format(folder_path, vehicle_type, key), "w")
    throttle_train_file.create_dataset(
        "segment", data=throttle_train[0:throttle_train_feature_num, :], dtype="float32")
    throttle_train_file.close()

    throttle_test_file = h5py.File(
        "{}/{}_throttle_test_{}.hdf5".format(folder_path, vehicle_type, key), "w")
    throttle_test_file.create_dataset(
        "segment", data=throttle_test[0:throttle_test_feature_num, :], dtype="float32")
    throttle_test_file.close()

    brake_train_file = h5py.File(
        "{}/{}_brake_train_{}.hdf5".format(folder_path, vehicle_type, key), "w")
    brake_train_file.create_dataset(
        "segment", data=brake_train[0:brake_train_feature_num, :], dtype="float32")
    brake_train_file.close()

    brake_test_file = h5py.File(
        "{}/{}_brake_test_{}.hdf5".format(folder_path, vehicle_type, key), "w")
    brake_test_file.create_dataset(
        "segment", data=brake_test[0:brake_test_feature_num, :], dtype="float32")
    brake_test_file.close()

    return feature_num
