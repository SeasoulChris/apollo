#!/usr/bin/env python

import numpy as np


def get_img_idx(local_point, local_center_idx, resolution):
    """
    Translate a point with respect to a local point on a integer img coordinates which as the upper
    left as origin

    Argument:
    local_point: a (2,) numpy array, the local point coordinates wrt local center
                 where y aixs point up
    local_center_idx: a (2,) integer numpy array, the local point idx
                      on the integer img coordinates
    resolution: a float representing the ratio of world coordianates resolution
                to the integer coordinates

    Return:
    local_point_idx: a (2,) integer numpy array, the point's indx on the integer coordinates
    """
    local_point_idx = np.round(local_point / resolution)
    local_point_idx = local_center_idx + (local_point_idx * np.array([1, -1]))
    return local_point_idx.astype(int)


def get_rotation_matrix(theta):
    """
    Return a rotation matrix rotating a 2d point ccw with angle theta in radian
    """
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def point_affine_transformation(point, local_center, theta):
    """
    Translate a point wrt local_center and rotate around the local center

    Arguments:
    point: a (2,) numpy array, point to be transformed
    local_center: a (2,) numpy array, point to be translated to and rotated around
    theta: ccw angle in radian around the local center to rotate the point

    Return:
    affined_point: transformed point
    """
    affined_point = point - local_center
    affined_point = np.dot(get_rotation_matrix(theta), affined_point.T).T
    return affined_point


def box_affine_tranformation(east_oriented_corner_points, box_center_point,
                             box_theta, local_center, theta, local_center_idx, resolution):
    """
    Transform a box wrt to a local center

    Arguments:
    east_oriented_corner_points: a (4, 2) numpy array, corner points coordinates around box center
    as origin box_center_point: a (2, ) numpy array, box center point wrt world frame
    box_theta: radian angle, box rotation angle from east wrt world frame
    local_center: a (2,) numpy array, point to be translated to and rotated around
    theta: ccw angle in radian around the local center to rotate the point

    Return:
    affine_box: transformed numpy array
    """
    rear_center_point = point_affine_transformation(
        box_center_point, local_center, theta)
    corner_points = [get_img_idx(point_affine_transformation(point.T,
                                                             np.array(
                                                                 [0, 0]),
                                                             box_theta)
                                 + rear_center_point,
                                 local_center_idx,
                                 resolution)
                     for point in east_oriented_corner_points]
    return np.asarray(corner_points)


def img_notblack_stacking(img, target_img):
    """
    Stack the not black pixel of a img onto the target image, imgs have to be in the same size

    Arguments:
    img: a 3 channel img in np.array
    target_img: the 3 channel img in np.array where the img is stacked onto

    Return:
    stacked_img: stacked 3 channel img
    """
    stacked_img = np.where(np.repeat(
        np.expand_dims(
            (img[:, :] == (0, 0, 0)).all(axis=2),
            axis=2),
        repeats=3,
        axis=2),
        target_img[:, :],
        img[:, :])
    return stacked_img


def img_replace_white_with_color(img, color):
    """
    replace the white pixels of a 3 channel img with the given color

    Arguments:
    img: img to be coloered at white pixel
    color: a size 3 tuple

    Return:
    color_img: colored img
    """
    colored_img = np.where(np.repeat(
        np.expand_dims(
            (img[:, :] == (255, 255, 255)).all(axis=2),
            axis=2),
        repeats=3,
        axis=2),
        color,
        img[:, :])
    return colored_img


def img_white_gradient_to_color_gradient(img, single_color_max_value):
    """
    replace the white gradient pixels(a, b, c where a == b == c) of a 3
        channel img with the given gradient single_color

    Arguments:
    img: img to be coloered at white pixel
    single_color_max_value: a size 3 numpy array, (0, 0, 255) for red max e.g

    Return:
    color_img: colored img
    """
    colored_img = np.where(
        np.repeat(
            np.expand_dims(
                np.logical_and(img[:, :, 0] == img[:, :, 1],
                               img[:, :, 0] == img[:, :, 2]),
                axis=2),
            repeats=3,
            axis=2),
        single_color_max_value * (img[:, :] / 255),
        img[:, :])
    return colored_img
