#!/usr/bin/env python

from collections import Counter
import glob
import random

import cv2 as cv
import scipy
from scipy.signal import filtfilt

from learning_algorithms.prediction.datasets.apollo_pedestrian_dataset.data_for_learning_pb2 \
    import *
from learning_algorithms.utilities.helper_utils import *
import learning_algorithms.prediction.datasets.apollo_pedestrian_dataset.data_for_learning_pb2


#######################################################################
# Helper Functions
#######################################################################
def point_to_idx(point_x, point_y):
    return (int(point_x/0.1 + 400), int(point_y/0.1 + 400))


def plot_img(future_pt, count, adjusted_future_pt=None):
    # black background
    img = np.zeros([1000, 800, 3], dtype=np.uint8)
    # draw boundaries
    cv.circle(img, (400, 400), 2, color=[255, 255, 255], thickness=4)
    cv.line(img, (0, 0), (799, 0), color=[255, 255, 255])
    cv.line(img, (799, 0), (799, 999), color=[255, 255, 255])
    cv.line(img, (0, 999), (0, 0), color=[255, 255, 255])
    cv.line(img, (799, 999), (0, 999), color=[255, 255, 255])
    for ts in range(future_pt.shape[0]):
        cv.circle(img, point_to_idx(future_pt[ts][0] - future_pt[0][0], future_pt[ts]
                                    [1] - future_pt[0][1]), radius=3, thickness=2, color=[0, 128, 128])
    if adjusted_future_pt is not None:
        for ts in range(adjusted_future_pt.shape[0]):
            cv.circle(img, point_to_idx(adjusted_future_pt[ts][0] - adjusted_future_pt[0][0], adjusted_future_pt[ts]
                                        [1] - adjusted_future_pt[0][1]), radius=3, thickness=2, color=[128, 0, 128])
    cv.imwrite('img={}.png'.format(count), cv.flip(cv.flip(img, 0), 1))


#######################################################################
# Coarse label cleaning; then smoothing to filter out useful data-points.
#######################################################################
def LabelCleaningCoarse(feature_seq, pred_len=30):
    # 1. Only keep pred_len length
    if len(feature_seq) < pred_len:
        return None
    obs_pos = np.array([[feature[0], feature[1]]
                        for feature in feature_seq[:pred_len]])
    obs_pos = obs_pos - obs_pos[0, :]
    # 2. Get the scalar acceleration, and angular speed of all points.
    obs_vel = (obs_pos[1:, :] - obs_pos[:-1, :]) / 0.1
    linear_vel = np.linalg.norm(obs_vel, axis=1)
    linear_acc = (linear_vel[1:] - linear_vel[0:-1]) / 0.1
    angular_vel = np.sum(
        obs_vel[1:, :] * obs_vel[:-1, :], axis=1) / ((linear_vel[1:] * linear_vel[:-1]) + 1e-6)
    turning_ang = (np.arctan2(
        obs_vel[-1, 1], obs_vel[-1, 0]) - np.arctan2(obs_vel[0, 1], obs_vel[0, 0])) % (2*np.pi)
    turning_ang = turning_ang if turning_ang < np.pi else turning_ang-2*np.pi
    # 3. Filtered the extream values for acc and ang_vel.
    if np.max(np.abs(linear_acc)) > 50:
        return None
    if np.min(angular_vel) < 0.85:
        return None
    # Get the statistics of the cleaned labels, and do some re-balancing to
    # maintain roughly the same distribution as before.
    if -np.pi/6 <= turning_ang <= np.pi/6:
        area = (obs_pos[0, 0]*obs_pos[1, 1] + obs_pos[1, 0]*obs_pos[-1, 1] + obs_pos[-1, 0]*obs_pos[0, 1]
                - obs_pos[0, 0]*obs_pos[-1, 1] - obs_pos[1, 0]*obs_pos[0, 1] - obs_pos[-1, 0]*obs_pos[1, 1])
        if area/(np.linalg.norm(obs_pos[1, :] - obs_pos[0, :]) + 1e-6) >= 3:
            return 'change_lane'
        else:
            return 'straight'
    elif -np.pi/2 <= turning_ang < -np.pi/6:
        return 'right'
    elif np.pi/6 < turning_ang <= np.pi/2:
        return 'left'
    else:
        return 'uturn'


def SmoothFeatureSequence(feature_seq):
    """
    feature_seq: a sequence of tuples (x, y, v_heading, v, length, width, timestamp, acc)
    """
    x_coords = []
    y_coords = []
    smoothed_feature_seq = []
    start_x = feature_seq[0][0]
    start_y = feature_seq[0][1]

    for feature in feature_seq:
        x_coords.append(feature[0] - start_x)
        y_coords.append(feature[1] - start_y)

    b, a = scipy.signal.butter(2, 0.72)
    smooth_x_coords = filtfilt(b, a, x_coords, method="gust")
    smooth_y_coords = filtfilt(b, a, y_coords, method="gust")

    for i in range(len(feature_seq)):
        smoothed_feature = list(feature_seq[i])
        smoothed_feature[0] = smooth_x_coords[i] + start_x
        smoothed_feature[1] = smooth_y_coords[i] + start_y
        smoothed_feature_seq.append(tuple(smoothed_feature))

    return smoothed_feature_seq


def LabelCleaningAndSmoothing(label_dir):
    label_dict_file_list = glob.glob(
        label_dir + '/**/future_status.npy', recursive=True)
    count = Counter()
    for label_dict_file in label_dict_file_list:
        label_dict = np.load(label_dict_file, allow_pickle=True).item()
        processed_label_dict = {}
        idx = 0
        for key, feature_seq in label_dict.items():
            pred_len = 30
            turn_type = LabelCleaningCoarse(feature_seq, pred_len)
            if turn_type:
                count[turn_type] += 1
                feature_seq = feature_seq[:pred_len]
                smoothed_feature_seq = SmoothFeatureSequence(feature_seq)
                processed_label_dict[key] = smoothed_feature_seq
                obs_pos = np.array([[feature[0], feature[1]]
                                    for feature in feature_seq])
                obs_pos = obs_pos - obs_pos[0, :]
                smoothed_obs_pos = np.array([[feature[0], feature[1]]
                                             for feature in smoothed_feature_seq])
                smoothed_obs_pos = smoothed_obs_pos - smoothed_obs_pos[0, :]
                # plot_img(obs_pos, idx, smoothed_obs_pos)
                # idx += 1
        print("Got " + str(len(processed_label_dict.keys())) +
              "/" + str(len(label_dict.keys())) + " labels left!")
        print(count)
        # np.save(label_dict_name.replace('future_status.npy',
        #                                 'processed_label.npy'), processed_label_dict)


#######################################################################
# Fine label cleaning; no smoothing.
#######################################################################
def LabelCleaningFine(feature_dir, label_dir, pred_len=30):
    # From feature_dir, locate those labels of interests.
    label_dict_list = glob.glob(label_dir + '/**/cleaned_label.npy', recursive=True)

    # Go through all labels of interests, filter out those noisy ones and
    # only retain those clean ones.
    count = Counter()
    file_count = 0
    for label_dict_name in label_dict_list:
        file_count += 1
        print ('Processing {}/{}'.format(file_count, len(label_dict_list)))
        label_dict = np.load(label_dict_name).item()
        cleaned_label_dict = {}
        idx = 0
        for key, feature_seq in label_dict.items():
            # 1. Only keep pred_len length
            if len(feature_seq) < pred_len:
                continue
            obs_pos = np.array([[feature[0], feature[1]] for feature in feature_seq[:pred_len]])
            obs_pos = obs_pos - obs_pos[0, :]
            # 2. Get the scalar acceleration, and angular speed of all points.
            obs_vel = (obs_pos[1:, :] - obs_pos[:-1, :]) / 0.1
            linear_vel = np.linalg.norm(obs_vel, axis=1)
            linear_acc = (linear_vel[1:] - linear_vel[0:-1]) / 0.1
            angular_vel = np.sum(obs_vel[1:, :] * obs_vel[:-1, :], axis=1) / \
                ((linear_vel[1:] * linear_vel[:-1]) + 1e-6)
            turning_ang = (np.arctan2(obs_vel[-1, 1], obs_vel[-1, 0]) -
                           np.arctan2(obs_vel[0, 1], obs_vel[0, 0])) % (2*np.pi)
            turning_ang = turning_ang if turning_ang < np.pi else turning_ang-2*np.pi
            # 3. Filtered the extream values for acc and ang_vel.
            if np.max(np.abs(linear_acc)) > 80:
                continue
            if np.min(angular_vel) < 0.8:
                continue
            # plot_img(obs_pos, idx)
            # print(idx, key)
            # idx += 1

            # Get the statistics of the cleaned labels, and do some re-balancing to
            # maintain roughly the same distribution as before.
            if -np.pi/6 <= turning_ang <= np.pi/6:
                if np.min(angular_vel) < 0.9 or np.max(np.abs(linear_acc)) > 30:
                    continue
                area = (obs_pos[0, 0]*obs_pos[1, 1] + obs_pos[1, 0]*obs_pos[-1, 1] + obs_pos[-1, 0]*obs_pos[0, 1]
                        - obs_pos[0, 0]*obs_pos[-1, 1] - obs_pos[1, 0]*obs_pos[0, 1] - obs_pos[-1, 0]*obs_pos[1, 1])
                if area/(np.linalg.norm(obs_pos[1, :] - obs_pos[0, :]) + 1e-6) >= 3:
                    count['change_lane'] += 1
                else:
                    count['straight'] += 1
            elif -np.pi/2 <= turning_ang < -np.pi/6:
                count['right'] += 1
            elif np.pi/6 < turning_ang <= np.pi/2:
                if np.max(np.abs(linear_acc)) > 30:
                    continue
                count['left'] += 1
            else:
                count['uturn'] += 1
            cleaned_label_dict[key] = feature_seq[:pred_len]

        print("Got " + str(len(cleaned_label_dict.keys())) +
              "/" + str(len(label_dict.keys())) + " labels left!")
        print(count)
        np.save(label_dict_name.replace('cleaned_label.npy', 'cleaner_label.npy'), cleaned_label_dict)
    print(count)
    return


def LabelBalance(label_dir, straight_remain_rate=0.25):
    label_dict_file_list = glob.glob(
        label_dir + '/**/future_status_clean.npy', recursive=True)
    count = Counter()
    for label_dict_file in label_dict_file_list:
        label_dict = np.load(label_dict_file, allow_pickle=True).item()
        processed_label_dict = {}
        idx = 0
        for key, feature_seq in label_dict.items():
            pred_len = 30
            turn_type = LabelCleaningCoarse(feature_seq, pred_len)
            if turn_type:
                select = True
                if turn_type == 'straight':
                    chance = random.uniform(0, 1)
                    if chance > straight_remain_rate:
                        continue
                count[turn_type] += 1
                feature_seq = feature_seq[:pred_len]
                processed_label_dict[key] = feature_seq

        print("Got " + str(len(processed_label_dict.keys())) +
              "/" + str(len(label_dict.keys())) + " labels left!")
        print(count)
        np.save(label_dict_file.replace('future_status_clean.npy',
                                        'future_status_clean_balance.npy'), processed_label_dict)


#######################################################################
# Main function.
#######################################################################
if __name__ == '__main__':
 #    # Option-a. First coarse label cleaning, then smoothing.
 #    LabelCleaningAndSmoothing('/data/labels-future-points/')
 #    # Option-b. Only fine label cleaning, no smoothing.
 #    LabelCleaningFine('test', '/home/jiacheng/work/apollo/data/apollo_vehicle_trajectory_data/labels-future-points-clean')
 #    LabelBalance('/home/xukecheng/labels', 0.21)
