#!/usr/bin/env python

import os
import sys

import numpy as np

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


def get_all_paths(label_path):
    """
    From label path to get both image directory path and
    camera calibration directory path as well.
    Output: (label_path, image_dir, calib_dir)
    """
    id, label_path = label_path
    label_dir, file_name = os.path.split(label_path)
    file_name, _ = os.path.splitext(file_name)
    data_dir, _ = os.path.split(label_dir)

    image_dir = os.path.join(os.path.join(data_dir, "images"))
    calib_dir = os.path.join(os.path.join(data_dir, "calib"))
    return (id, (label_path, image_dir, calib_dir))


def process_data(paths):
    """
    Read data from paths and preprocss to get input and
    ground truth outputs.
    """
    id, paths = paths
    label_path, image_dir, calib_dir = paths
    image_data, y_true, cls_box_map, objs, calib = \
        process_label_file(label_path,
                           image_dir,
                           calib_dir,
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
    return (id, (image_data, y_true, cls_box_map, objs, calib))


def filter_classes(element):
    """
    Filter out classes that are not to be considered for
    training.
    Input element: should be the output of function 
      'process_data'.
    """
    id, element = element
    image_data, y_true, cls_box_map, objs, calib = element
    if CLS_TO_CONSIDER is not None:
        # Zero out elements in y_true
        for i, scale_label in enumerate(y_true):
            bat, cell_rows, cell_cols, anchors, cls = \
                np.where(scale_label[..., 5:5+cfg.num_classes])

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
    return (id, (image_data, y_true, [cls_box_map], [objs], [calib]))
