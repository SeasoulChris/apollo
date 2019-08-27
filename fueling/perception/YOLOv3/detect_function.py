#!/usr/bin/env python

import numpy as np
import tensorflow as tf


def yolo_head(feature_maps, anchors, num_classes,
              num_angle_bins, input_shape, calc_loss=False):
    """
    Convert final layer features to bounding box parameters.
    (Features learned by the convolutional layers --->
        a classifier/regressor which makes the detection prediction)
    :param feature_maps: the feature maps(3 scales) learned by the convolutional layers
    :param anchors: num of anchors for each scale shape=(k,2)
    :param num_classes: integer
    :param input_shape: shape of input tensor to the network
    :return: 
        grid: (cel_row, cel_col, 1, 2)
        feature_maps_reshape: (bs, cel_row, cel_col, anchors_per_box, 10+num_cls)
        box_xy:  [bs, cel_row, cel_col, anchors_per_box, 2], 2: x,y center point of BB
        box_wh:  [bs, cel_row, cel_col, anchors_per_box, 2], 2: w,h
        box_confidence:  [bs, cel_row, cel_col, anchors_per_box, 1], 1: conf
        box_class_probs:  [bs, cel_row, cel_col, anchors_per_box, num_cls], prob of each class
        box_cs:  [bs, cel_row, cel_col, anchors_per_box, 2], cos(alpha), sin(alpha)
        box_hwl:  [bs, cel_row, cel_col, anchors_per_box, 3], 3d hwl
    """
    num_anchors = len(anchors)  
    # Reshape to batch, height, width, num_anchors, box_params
    # (anchors_per_cell, 2)
    anchors_tensor = tf.cast(anchors, dtype=feature_maps.dtype)
    # shape=[1,1,1,anchors_per_cell,2]
    anchors_tensor = tf.reshape(anchors_tensor, [1, 1, 1, num_anchors, 2])

    bs = tf.shape(feature_maps)[0]
    # CREATE A GRID FOR EACH SCALE
    with tf.name_scope('Create_GRID'):
        grid_shape = tf.shape(feature_maps)[1:3]  # height, width
        grid_y = tf.range(0, grid_shape[0])
        grid_x = tf.range(0, grid_shape[1])
        # shape=([cel_row,  1,  1,  1])
        grid_y = tf.reshape(grid_y, [-1, 1, 1, 1])
        # [1, cel_col, 1, 1]
        grid_x = tf.reshape(grid_x, [1, -1, 1, 1])
        # [cel_row, 1, 1, 1] ---> [cel_row, cel_col, 1, 1]
        grid_y = tf.tile(grid_y, [1, grid_shape[1], 1, 1])
        # [1, cel_col, 1, 1] ---> [cel_row, cel_col, 1, 1]
        grid_x = tf.tile(grid_x, [grid_shape[0], 1, 1, 1])
        # (cel_row, cel_col,  1,  2)
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, dtype=feature_maps.dtype)

        # (bs, 1, 1, 1)
        bs_idx = tf.reshape(tf.range(0, bs), [-1, 1, 1, 1])
        # (1, cel_row, 1, 1)
        row_idx = tf.reshape(tf.range(0, grid_shape[0]), [1, -1, 1, 1])
        # (1, 1, cel_col, 1)
        col_idx = tf.reshape(tf.range(0, grid_shape[1]), [1, 1, -1, 1])
        # (1, 1, 1, num_anchors)
        anc_idx = tf.reshape(tf.range(0, num_anchors), [1, 1, 1, -1])
        # (bs, cel_row, cel_col, num_anchors)
        bs_row_col_anc = bs_idx * grid_shape[0] * grid_shape[1] * num_anchors + \
            row_idx * grid_shape[1] * num_anchors + col_idx * num_anchors + anc_idx
        
    feature_xy_reshape = tf.reshape(tf.concat([feature_maps[:, :, :, s * 4:s * 4 + 2]
                                               for s in range(num_anchors)], axis=-1), 
                              [bs, grid_shape[0], grid_shape[1], num_anchors, 2])
    feature_wh_reshape = tf.reshape(tf.concat([feature_maps[:, :, :, s * 4 + 2:s * 4 + 4]
                                               for s in range(num_anchors)], axis=-1), 
                              [bs, grid_shape[0], grid_shape[1], num_anchors, 2])
    feature_conf_reshape = tf.reshape(feature_maps[:, :, :, num_anchors * (4):num_anchors * (5)], 
                                [bs, grid_shape[0], grid_shape[1], num_anchors, 1])
    feature_cls_reshape = tf.reshape(feature_maps[:, :, :, 
                                                num_anchors * 5:num_anchors * (5 + num_classes)], 
                               [bs, grid_shape[0], grid_shape[1], num_anchors, num_classes])
    feature_hwl_reshape = tf.reshape(feature_maps[:, :, :,
                             num_anchors * (5 + num_classes):num_anchors * (5 + num_classes * 4)], 
                               [bs, grid_shape[0], grid_shape[1], num_anchors, num_classes * 3])

    channel_idx_begin = num_anchors * (5 + num_classes * 4)
    channel_idx_end = num_anchors * (5 + num_classes * 4 + num_angle_bins)
    feature_cs_conf_reshape = \
          tf.reshape(feature_maps[:, :, :, channel_idx_begin:channel_idx_end],
                     [bs, grid_shape[0], grid_shape[1], num_anchors, num_angle_bins])

    channel_idx_begin = num_anchors * (5 + num_classes * 4 + num_angle_bins)
    channel_idx_end = num_anchors * (5 + num_classes * 4 + num_angle_bins * 3)
    feature_cs_reshape = \
        tf.reshape(feature_maps[:, :, :, channel_idx_begin:channel_idx_end], 
                   [bs, grid_shape[0], grid_shape[1], num_anchors, num_angle_bins * 2])

    with tf.name_scope('top_feature_maps'):
        # ======================= Get feature values ==========================
        # (bs, cel_row, cel_col, anchors_per_cel, 2)
        box_xy = tf.sigmoid(feature_xy_reshape)
        # (bs, cel_row, cel_col, anchors_per_cel, 2)
        box_wh = tf.exp(tf.clip_by_value(feature_wh_reshape, -10, 5))
        # (bs, cel_row, cel_col, anchors_per_cell, 1)
        box_confidence = tf.sigmoid(feature_conf_reshape)
        # for 3D
        # (bs, cel_row, cel_col, anchors_per_cel, num_cls)
        box_class_probs = tf.nn.softmax(feature_cls_reshape)
        box_hwl = feature_hwl_reshape
        box_cs_conf = tf.nn.softmax(feature_cs_conf_reshape)
        # (bs, cel_row, cel_col, anchors_per_cel, num_angle_bins, 2)
        box_cs = tf.nn.l2_normalize(
                   tf.reshape(feature_cs_reshape,
                       [bs, grid_shape[0], grid_shape[1], num_anchors, num_angle_bins, 2]),
                   dim=-1)
        
        # ====================== Convert to usable scales =========================
        # Adjust predictions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        # (x,y + grid)/n. ---> in between (0., 1.)
        # (bs, cel_row, cel_col, anchors_per_cell, 2)
        box_xy = (box_xy + grid) / tf.cast(grid_shape[::-1], dtype=feature_maps.dtype)
        # (bs, cel_row, cel_col, anchors_per_cel, 2)
        box_wh = box_wh * anchors_tensor / tf.cast(input_shape[::-1], dtype=feature_maps.dtype)

        # Grap the corresponding hwl from output
        # (bs, cel_row, cel_col, anchors_per_cel)
        c = tf.argmax(box_class_probs, axis=-1)
        # (bs, cel_row, cel_col, anchors_per_cel)
        flat_idx = tf.cast(bs_row_col_anc, dtype=tf.int32) * num_classes * 3 + \
                   tf.cast(c, dtype=tf.int32) * 3
        flat_idx = tf.reshape(flat_idx, (bs, grid_shape[0], grid_shape[1], num_anchors, 1))
        # (bs, cel_row, cel_col, anchors_per_cel, 3)
        flat_idx = tf.concat([flat_idx, flat_idx + 1, flat_idx + 2], axis=-1)
        # (bs*cel_row*cel_col*anchors_per_cel*3)
        flat_idx = tf.reshape(flat_idx, [-1])
        # (bs*cel_row*cel_col*anchors_per_cel*n)
        feature_flat = tf.reshape(box_hwl, [-1])
        box_hwl = tf.reshape(tf.gather(feature_flat, flat_idx),
                       [bs, grid_shape[0], grid_shape[1], num_anchors, 3])
        box_hwl = tf.exp(tf.clip_by_value(box_hwl, -10, 5))
      
        # Grap the corresponding cos, sin of alpha angle
        # (bs, cel_row, cel_col, anchors_per_cel)
        bins = tf.argmax(box_cs_conf, axis=-1)
        # (bs, cel_row, cel_col, anchors_per_cel)
        flat_idx_bin = tf.cast(bs_row_col_anc, dtype=tf.int32) * num_angle_bins * 2 + \
                       tf.cast(bins, dtype=tf.int32) * 2
        flat_idx_bin = tf.reshape(flat_idx_bin,
                            [bs, grid_shape[0], grid_shape[1], num_anchors, 1]) 
        # (bs, cel_row, cel_col, anchors_per_cel, 2)
        flat_idx_bin = tf.concat([flat_idx_bin, flat_idx_bin+1], axis=-1)
        # (bs*cel_row*cel_col*anchors_per_cel*2) 
        flat_idx_bin = tf.reshape(flat_idx_bin, [-1])
        feature_flat_cs = tf.reshape(box_cs, [-1])
        box_cs = tf.reshape(tf.gather(feature_flat_cs, flat_idx_bin),
                      [bs, grid_shape[0], grid_shape[1], num_anchors, 2])
        # Convert to obj local angle from angle offset
        # (bs, cel_row, cel_col, anchors, 1)
        box_atan = tf.atan2(box_cs[..., 1:], box_cs[..., 0:1])
        principle_angle = 2 * np.pi / num_angle_bins
        box_angle = box_atan + \
                    tf.reshape(tf.add(tf.scalar_mul(principle_angle, tf.to_float(bins)), 
                                    principle_angle/2),
                         [bs, grid_shape[0], grid_shape[1], num_anchors, 1])
        box_cs = tf.concat([tf.cos(box_angle), tf.sin(box_angle)], axis=-1)

    if calc_loss == True:
        return grid, \
               [feature_xy_reshape, 
                feature_wh_reshape, 
                feature_conf_reshape, 
                feature_cls_reshape, 
                feature_hwl_reshape, 
                feature_cs_conf_reshape, 
                feature_cs_reshape], \
               box_xy, box_wh, \
               box_cs_conf, \
               box_cs, \
               box_hwl
    return box_xy, box_wh, box_confidence, box_class_probs, box_cs_conf, box_cs, box_hwl





