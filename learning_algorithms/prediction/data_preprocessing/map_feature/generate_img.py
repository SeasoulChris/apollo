#!/usr/bin/env python

import argparse
import glob
import os

import numpy as np
import cv2 as cv

from modules.prediction.proto import offline_features_pb2

from learning_algorithms.prediction.data_preprocessing.map_feature.obstacle_mapping import ObstacleMapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate imgs for a folder of frame_env.x.bin')
    parser.add_argument('-i', '--input', type=str, help='input directory')
    parser.add_argument('-o', '--output', type=str, help='output directory')
    parser.add_argument('-r', '--region', type=str, default="san_mateo", help='image region')
    args = parser.parse_args()

    output_dir = args.output
    if os.path.isdir(output_dir):
        print(output_dir + " directory exists!")
    else:
        os.makedirs(output_dir)
        print("Making output directory: " + output_dir)

    obs_pos_dict = dict()
    list_input = glob.glob(args.input+'/frame_env.*.bin')
    print(list_input)
    for input_file in list_input:
        list_frame = offline_features_pb2.ListFrameEnv()
        with open(input_file, 'r') as file_in:
            list_frame.ParseFromString(file_in.read())
        print("Finish reading proto: " + input_file)
        for idx, frame_env in enumerate(list_frame.frame_env):
            try:
                obstacle_mapping = ObstacleMapping(args.region, frame_env)
                # print("Drawing frame " + str(idx) + "/" + str(len(list_frame.frame_env)))
                for history in frame_env.obstacles_history:
                    if not history.is_trainable:
                        continue
                    key = "{}@{:.3f}".format(history.feature[0].id, history.feature[0].timestamp)
                    img = obstacle_mapping.crop_by_history(history)
                    filename = "/" + key + ".png"
                    cv.imwrite(os.path.join(output_dir + filename), img)
                    obs_pos_dict[key] = [(feature.position.x, feature.position.y)
                                         for feature in history.feature]
                    # print("Writing to: " + os.path.join(output_dir + filename))
            except:
                print("Possible error on frame: " + str(idx) + "/" + str(len(list_frame.frame_env)))
    np.save(os.path.join(output_dir+"/obs_pos.npy"), obs_pos_dict)
