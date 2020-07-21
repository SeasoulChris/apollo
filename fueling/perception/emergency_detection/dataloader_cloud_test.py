#!/usr/bin/env python

from absl import flags

from fueling.common.base_pipeline import BasePipeline

import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class Yolo_dataset(Dataset):
    def __init__(self, lable_path, dataset_dir):
        super(Yolo_dataset, self).__init__()

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(j) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        img_path = self.imgs[index]
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)
        img_path = os.path.join(self.dataset_dir, img_path)
        img = cv2.imread(img_path)
        box = [1,2,3,4]
        return img, box



def train_yolov4():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    train_label_path = '/fuel/fueling/perception/emergency_detection/data/coins/train.txt'
    val_label_path = '/fuel/fueling/perception/emergency_detection/data/coins/val.txt'
    dataset_dir = '/fuel/fueling/perception/emergency_detection/data/coins/'
    '''

    train_label_path = '/mnt/bos/modules/perception/emergency_detection/data/coins/train.txt'
    val_label_path = '/mnt/bos/modules/perception/emergency_detection/data/coins/val.txt'
    dataset_dir = '/mnt/bos/modules/perception/emergency_detection/data/coins/'

    # ************************ read directly ****************************
    print('*********************************** read directly begin ***********************************')
    lable_path = '/mnt/bos/modules/perception/emergency_detection/data/coins/train.txt'
    truth = {}
    f = open(lable_path, 'r', encoding='utf-8')
    for line in f.readlines():
        data = line.split(" ")
        truth[data[0]] = []
        for i in data[1:]:
            truth[data[0]].append([int(j) for j in i.split(',')])

    imgs = list(truth.keys())
    print('read image list length: ', len(imgs))
    dataset_dir='/mnt/bos/modules/perception/emergency_detection/data/coins'
    img_path = os.path.join(dataset_dir, imgs[0])
    img = cv2.imread(img_path)
    print('read image, size: ', img.shape)
    print('*********************************** read directly end ***********************************')



    # ************************ read by DataLoader ****************************
    train_dataset = Yolo_dataset(train_label_path, dataset_dir)
    val_dataset = Yolo_dataset(val_label_path, dataset_dir)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)
    #val_loader = DataLoader(val_dataset, batch_size=1 // 1, shuffle=True, num_workers=8,pin_memory=True, drop_last=True, collate_fn=val_collate)

    print('*********************************** DataLoader begin ***********************************')
    epochs = 2
    for epoch in range(epochs):
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                print(len(batch))

    print('*********************************** DataLoader end ***********************************')



class EmergencyVehicleDetector(BasePipeline):
    """Demo pipeline."""

    def run(self):
        #train_yolov4()
        self.to_rdd(range(1)).foreach(self.train)

    @staticmethod
    def train(instance_id):
        train_yolov4()

if __name__ == '__main__':
    EmergencyVehicleDetector().main()
    #train_yolov4()
