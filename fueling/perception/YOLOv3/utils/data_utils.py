#!/usr/bin/env python

import os
import re
import sys

import glob
import numpy as np
import random

from fueling.perception.YOLOv3.utils.yolo_utils import process_label_file
import fueling.perception.YOLOv3.config as cfg


INPUT_SHAPE = [cfg.Input_height, cfg.Input_width]
ANCHORS = cfg.anchors
CLS_NAME_ID_MAP = cfg.class_map
NUM_CLASSES = cfg.num_classes
NUM_ANGLE_BINS = cfg.num_angle_bins
BIN_OVERLAP_FRAC = cfg.bin_overlap_fraction
NUM_OUTPUT_LAYERS = cfg.num_output_layers
RANDOM_COLOR_SHIFT = cfg.random_color_shift
COLOR_SHIFT_PERCENTAGE = cfg.color_shift_percentage
RANDOM_CROP = cfg.random_crop
RANDOM_FLIP = cfg.random_flip
FLIP_CHANCE = cfg.flip_chance
RANDOM_JITTER = cfg.random_jitter
JITTER_CHANCE = cfg.jitter_chance
JITTER_PERCENTAGE = cfg.jitter_percentage
CLS_TO_CONSIDER = cfg.classes_to_consider


def get_all_image_paths(dataset_path, sample=0):
    image_dir = os.path.join(dataset_path, "images_all")
    image_path_list = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_path_list += glob.glob(os.path.join(image_dir, "*.png"))
    train_data = []
    test_data = []
    if sample:
        random.shuffle(image_path_list)
        idx = int(0.8 * len(image_path_list))
        train_data = image_path_list[:idx]
        test_data = image_path_list[idx:]
        return (train_data, test_data)
    else:
        return image_path_list


def get_all_paths(image_path):
    """
    From label path to get both image directory path and
    camera calibration directory path as well.
    Output: (label_path, image_dir, calib_dir)
    """
    image_path = image_path
    image_dir, file_name = os.path.split(image_path)
    file_name, _ = os.path.splitext(file_name)
    data_dir, _ = os.path.split(image_dir)

    label_path = os.path.join(os.path.join(data_dir, "label_all"), "{}.txt".format(file_name))
    calib_path = os.path.join(os.path.join(data_dir, "calib_all"), "{}.txt".format(file_name))
    return (label_path, image_path, calib_path)


def process_data(paths):
    """
    Read data from paths and preprocss to get input and
    ground truth outputs.
    """
    label_path, image_path, calib_path = paths
    image_data, y_true, cls_box_map, objs, calib, original_image = \
        process_label_file(label_path,
                           image_path,
                           calib_path,
                           input_shape=INPUT_SHAPE,
                           anchors=ANCHORS,
                           cls_name_id_map=CLS_NAME_ID_MAP,
                           num_classes=NUM_CLASSES,
                           num_angle_bins=NUM_ANGLE_BINS,
                           bin_overlap_frac=BIN_OVERLAP_FRAC,
                           num_output_layers=NUM_OUTPUT_LAYERS,
                           random_color_shift=RANDOM_COLOR_SHIFT,
                           color_shift_percentage=COLOR_SHIFT_PERCENTAGE,
                           random_crop_=RANDOM_CROP,
                           random_flip=RANDOM_FLIP,
                           flip_chance=FLIP_CHANCE,
                           random_jitter_=RANDOM_JITTER,
                           jitter_chance=JITTER_CHANCE,
                           jitter_percentage=JITTER_PERCENTAGE)
    return (image_data, y_true, cls_box_map, objs, calib, original_image)


def filter_classes(element):
    """
    Filter out classes that are not to be considered for
    training.
    Input element: should be the output of function
      'process_data'.
    """
    image_data, y_true, cls_box_map, objs, calib, original_image = element
    if CLS_TO_CONSIDER is not None:
        # Zero out elements in y_true
        for i, scale_label in enumerate(y_true):
            bat, cell_rows, cell_cols, anchors, cls = \
                np.where(scale_label[..., 5:5 + cfg.num_classes])

            mask = ~np.isin(cls, cfg.classes_to_consider)
            y_true[i][bat[mask], cell_rows[mask],
                      cell_cols[mask], anchors[mask]] = 0.0

        # Pop elements out of cls_box_map.
        keys = []
        for key, value in cls_box_map.items():
            # cls_box_map: class_id to a list of 2d boxes
            # [xmin, ymin, xmax, ymax]
            if key not in CLS_TO_CONSIDER:
                keys.append(key)
        for key in keys:
            cls_box_map.pop(key)
    return (image_data, y_true, cls_box_map, objs, calib, original_image)


def get_latest_model(model_folder_path, model_name_prefix):
    """Get the path of latest model from given folder"""
    latest_idx, latest_model = 0, None
    for file_name in os.listdir(model_folder_path):
        se = re.search(r'^{}-(\d+)\.data-[\S]*'.format(model_name_prefix), file_name, re.M | re.I)
        if se and int(se.group(1)) > latest_idx:
            latest_idx = int(se.group(1))
            latest_model = file_name
    return latest_model

def get_lastest_step(model_folder_path, model_name_prefix):
    """Get the latest step of existing training"""
    lastest_model = get_latest_model(model_folder_path, model_name_prefix)
    if not latest_model:
        return 0 
    return (int)(latest_model.split('-')[1].split('.')[0])

def get_restore_file_path(model_folder_path, model_name_prefix):
    """Get restore path_filename_prefix that satifies the TF restore method"""
    latest_model_file = get_latest_model(model_folder_path, model_name_prefix)
    if latest_model_file:
        return os.path.join(model_folder_path, latest_model_file.split('.')[0])
    return None

