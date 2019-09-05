#!/usr/bin/env python

import cv2
import numpy as np
import random


def random_hsv_shift(img, low=0.8, high=1.2):
    """
    Randomly change saturation and v in hsv color space.
    Input:
    img: a numpy array image in RGB space.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = (np.minimum(np.floor(hsv[:, :, 1] * random.uniform(low, high)), 255))\
        .astype(dtype=np.uint8)
    hsv[:, :, 2] = (np.minimum(np.floor(hsv[:, :, 2] * random.uniform(low, high)), 255))\
        .astype(dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_crop(img, low=0.8):
    """
    Randomly crop the image up to the range of (low, 1) of the original image size.
    Input:
    img: a numpy array image
    """
    crop_scale = random.uniform(0.8, 1)
    crop_width = np.floor(img.shape[1] * crop_scale).astype(np.int32)
    crop_height = np.floor(img.shape[0] * crop_scale).astype(np.int32)

    x = random.randint(0, img.shape[1] - crop_width)
    y = random.randint(0, img.shape[0] - crop_height)
    return img[y:y + crop_height, x:x + crop_width], x, y, crop_width, crop_height


def random_crop_bbox(box, start_x, start_y, crop_width, crop_height):
    """
    Crop a bounding box according to the crop of the image.
    Input:
    box: a numpy array of box [x0, y0, x1, y1]
    start_x: the top_left x of the crop.
    start_y: the top_left y of the crop.
    crop_width: the crop wdith.
    crop_height: the crop height.
    Return:
    None if the bbox is invalid after cropping.
    list -> [x0, y0, x1, y1] of the new cropped bbox.
    """
    x0, y0, x1, y1 = box
    if x1 < start_x or y1 < start_y or \
       x0 > start_x + crop_width - 1 or y0 > start_y + crop_height - 1:
        return None

    x0 = max(start_x, x0) - start_x
    y0 = max(start_y, y0) - start_y
    x1 = min(start_x + crop_width - 1, x1) - start_x
    y1 = min(start_y + crop_height - 1, y1) - start_y

    return [x0, y0, x1, y1]


def random_jitter(box, jitter=0.1):
    """
    Randomly jitter the bbounding box.
    Input:
    box: a numpy array of box [x0, y0, x1, y1]
    jitter: upper bound of box jittering.
    Return:
    The jittered box [x0, y0, x1, y1], note they may go below 0 or beyond dimensions of image
    """
    x0, y0, x1, y1 = box
    jitter_width = (x1 - x0 + 1) * jitter
    jitter_height = (y1 - y0 + 1) * jitter

    x0_var = random.uniform(-jitter_width / 2.0, jitter_width / 2.0)
    y0_var = random.uniform(-jitter_height / 2.0, jitter_height / 2.0)
    x1_var = random.uniform(-jitter_width / 2.0, jitter_width / 2.0)
    y1_var = random.uniform(-jitter_height / 2.0, jitter_height / 2.0)
    return [x0 + x0_var, y0 + y0_var, x1 + x1_var, y1 + y1_var]


def flip_image_lr(image):
    """
    Flip am image left to right.
    Input:
    image: a numpy RGB image.
    """
    return np.fliplr(image)


def flip_bbox_lr(bbox, width):
    """
    Flip a bbox left to right.
    Input:
    bbox: a list,  [x0, y0, x1, y1]
    width: width of the image.
    height: height of the image.
    """
    x0, y0, x1, y1 = bbox
    new_x0 = width - x0 - (x1 - x0)
    x1 = width - x1 + (x1 - x0)
    return [new_x0, y0, x1, y1]
