#!/usr/bin/env python

import cv2 as cv
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import fueling.common.logging as logging

from fueling.common.coord_utils import CoordUtils
import fueling.common.file_utils as file_utils
from learning_algorithms.prediction.data_preprocessing.map_feature.online_mapping import ObstacleMapping

MAP_IMG_DIR = "/fuel/learning_algorithms/prediction/data_preprocessing/map_feature/"
ENABLE_IMG_DUMP = False

class SemanticMapDataset(Dataset):
    def __init__(self, data_dir):
        self.map_region = []
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.instances = []

        self.reference_world_coord = []

        accumulated_data_pt = 0

        # TODO(Hongyi): add the drawing class here.
        self.base_map = {
            "sunnyvale": cv.imread(MAP_IMG_DIR + "sunnyvale_with_two_offices.png"),
            "san_mateo": cv.imread(MAP_IMG_DIR + "san_mateo.png")}

        logging.info('Processing directory: {}'.format(data_dir))
        all_file_paths = file_utils.list_files(data_dir)
        # sort by filenames numerically: learning_data.<int>.bin.training_data.npy
        all_file_paths.sort(
            key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        for file_path in all_file_paths:
            if 'training_data' not in file_path:
                continue
            logging.info("loading {} ...".format(file_path))
            file_content = np.load(file_path, allow_pickle=True).tolist()
            self.instances += file_content

        self.total_num_data_pt = len(self.instances)
        logging.info('Total number of data points = {}'.format(self.total_num_data_pt))

    def __len__(self):
        return self.total_num_data_pt

    def __getitem__(self, idx):
        region = 'sunnyvale'
        world_coord = self.instances[idx][0][0:2]
        world_coord += [self.instances[idx][0][3]]  # heading
        # logging.info('world coord:{}'.format(world_coord))
        adc_mapping = ObstacleMapping(region, self.base_map[region], world_coord, None)

        adc_pose = [world_coord[0], world_coord[1]]
        img = adc_mapping.crop_by_rectangle(adc_pose)

        if self.img_transform:
            img = self.img_transform(img)

        if ENABLE_IMG_DUMP:
            cv.imwrite("/fuel/data/tmp/img{}.png".format(idx), img)

        # print("features:")
        # print(self.instances[idx][0])

        # print("label:")

        # print(self.instances[idx][1])
        # return ((img, self.instances[idx][0]),
        #          self.instances[idx][1])
        return ((img,
                torch.from_numpy(np.asarray(self.instances[idx][0])).float()),
                torch.from_numpy(np.asarray(self.instances[idx][1])).float())

if __name__ == '__main__':
    # Given cleaned labels, preprocess the data-for-learning and generate
    # training-data ready for torch Dataset.

    # dump one instance image for debug
    dataset = SemanticMapDataset('/fuel/fueling/planning/datasets/training')
    # dataset[0]
    dataset[100]
