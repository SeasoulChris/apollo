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

import argparse
import glob
import os
import numpy as np
import cv2 as cv

from modules.prediction.proto import offline_features_pb2
from obstacle_mapping import ObstacleMapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate imgs for a folder of frame_env.x.bin')
    parser.add_argument('-i', '--input', type=str, help='input directory')
    parser.add_argument('-o', '--output', type=str, help='output directory')
    args = parser.parse_args()

    output_dir = args.output
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists!")
    else:
		os.mkdir(output_dir)
		print("Making output directory: " + output_dir)

    obs_pos_dict = dict()
    list_input = glob.glob(args.input+'/frame_env.*.bin')
    print(list_input)
    for input_file in list_input:
        list_frame = offline_features_pb2.ListFrameEnv()
        with open(input_file, 'r') as file_in:
            list_frame.ParseFromString(file_in.read())
        print("Finish reading proto: " + input_file)
        for frame_env in list_frame.frame_env:
            obstacle_mapping = ObstacleMapping("san_mateo", frame_env)
            for history in frame_env.obstacles_history:
                if not history.is_trainable:
                    continue
                key = "{}@{:.3f}".format(history.feature[0].id, history.feature[0].timestamp)
                filename = key + ".png"
                obs_pos = []
                for feature in history.feature:
                    obs_pos.append((feature.position.x, feature.position.y))
                obs_pos_dict[key] = obs_pos
                img = obstacle_mapping.crop_by_history(history)
                cv.imwrite(output_dir + filename, img)
    np.save(output_dir+"obs_pos.npy", obs_pos_dict)
