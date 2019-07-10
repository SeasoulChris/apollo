###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

from collections import Counter
import cv2 as cv
import glob
import numpy as np


def point_to_idx(point_x, point_y):
    return (int(point_x/0.1 + 400), int(point_y/0.1 + 400))


def plot_img(future_pt, count):
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
    cv.imwrite('img={}.png'.format(count), cv.flip(cv.flip(img, 0), 1))


def LabelCleaning(feature_dir, label_dir, pred_len=30):
    # From feature_dir, locate those labels of interests.
    label_dict_list = glob.glob(
        label_dir + '/**/cleaned_label.npy', recursive=True)

    # Get the statistics of all the labels. (histogram of how many left-turns,
    # right-turns, u-turns, go-straight, etc.)

    # Go through all labels of interests, filter out those noisy ones and
    # only retain those clean ones.
    count = Counter()
    for label_dict_name in label_dict_list:
        label_dict = np.load(label_dict_name, allow_pickle=True).item()
        cleaned_label_dict = {}
        idx = 0
        for key, feature_seq in label_dict.items():
            # 1. Only keep pred_len length
            if len(feature_seq) < pred_len:
                continue
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
                continue
            if np.min(angular_vel) < 0.85:
                continue
            # plot_img(obs_pos, idx)
            # print(idx, key)
            # idx += 1
            # Get the statistics of the cleaned labels, and do some re-balancing to
            # maintain roughly the same distribution as before.
            if -np.pi/6 <= turning_ang <= np.pi/6:
                if np.max(np.abs(linear_acc)) > 30 or np.min(angular_vel) < 0.9:
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
            # 4. Those with large residual errors should be removed.
            cleaned_label_dict[key] = feature_seq[:pred_len]
        print("Got " + str(len(cleaned_label_dict.keys())) +
              "/" + str(len(label_dict.keys())) + " labels left!")
        print(count)
        np.save(label_dict_name.replace('cleaned_label.npy',
                                        'future_status.npy'), cleaned_label_dict)
    return


if __name__ == '__main__':
    LabelCleaning('test', '/home/sunhongyi/Downloads/labels-future-points/')
