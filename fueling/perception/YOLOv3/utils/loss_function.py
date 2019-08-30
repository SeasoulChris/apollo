#!/usr/bin/env python

import math

import keras.backend as K
import tensorflow as tf

from fueling.perception.YOLOv3.detect_function import yolo_head
import fueling.perception.YOLOv3.config as cfg


def convert_raw_output_to_box(yolo_outputs, anchors):
    """
    :param YOLO_outputs: list of 3 elements,
           3*[(bs, cel_row, cel_col, anchors_per_cel*(10+num_cls)]
    :param anchors: array, shape=(T, 2), wh
    """
    anchor_mask = cfg.anchor_mask

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(yolo_outputs[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(yolo_outputs[0]))
                   for l in range(3)]

    bs = K.shape(yolo_outputs[0])[0]  # batch size, tensor

    xy_wh_conf = []
    for l in range(3):
        # get YOLO output
        pred_xy, pred_wh, box_confidence, box_class_probs, \
            pred_cs_conf, pred_cs, pred_hwl = yolo_head(yolo_outputs[l],
                                                        anchors[anchor_mask[l]],
                                                        cfg.num_classes,
                                                        cfg.num_angle_bins,
                                                        input_shape,
                                                        calc_loss=False)
        pred_x1 = pred_xy[..., 0:1] - pred_wh[..., 0:1] / 2
        pred_y1 = pred_xy[..., 1:2] - pred_wh[..., 1:2] / 2
        pred_x2 = pred_xy[..., 0:1] + pred_wh[..., 0:1] / 2
        pred_y2 = pred_xy[..., 1:2] + pred_wh[..., 1:2] / 2

        xy_wh_conf.append(tf.concat([pred_x1, pred_y1,
                                     pred_x2, pred_y2,
                                     box_confidence,
                                     pred_cs, pred_hwl,
                                     box_class_probs], axis=-1))

    xy_wh_conf = \
        tf.concat(
            [tf.reshape(e,
                        shape=(-1,
                               len(anchor_mask[l]) * K.shape(yolo_outputs[l])[1] *
                               K.shape(yolo_outputs[l])[2],
                               10 + cfg.num_classes))
             for l, e in enumerate(xy_wh_conf)],
            axis=1)

    return xy_wh_conf


def compute_loss(yolo_outputs, y_true, anchors, num_classes,
                 num_angle_bins, ignore_thresh, print_loss=False):
    """
    Return yolo_loss tensor
    :param YOLO_outputs: list of 3 elements,
            3*[(bs, cel_row, cel_col, anchors_per_cel*(10+num_cls)]
    :param Y_true: list(3 array) [(bs,h//32,w//32,anchors_per_cel, 10+numclass),
                                  (bs,h//16,w//16,anchors_per_cel, 10+numclass),
                                  (bs,h//8,w//8, anchors_per_cel, 10+numclass)]
    :param anchors: array, shape=(T, 2), wh
    :param num_classes: 
    :param ignore_thresh:float, the iou threshold whether to ignore object confidence loss
    :return: loss
    """
    anchor_mask = cfg.anchor_mask

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(3)]
    bs = tf.shape(yolo_outputs[0])[0]  # batch size, tensor
    bs = tf.cast(bs, tf.int32)

    # (bs, 1, 1, 1)
    bs_idx = tf.cast(tf.reshape(tf.range(0, bs), [-1, 1, 1, 1]), dtype=tf.int32)

    scale_loss_list = []
    xy_loss_list = []
    wh_loss_list = []
    positive_conf_loss_list = []
    negative_conf_loss_list = []
    class_loss_list = []
    alpha_loss_list = []
    hwl_loss_list = []
    for l in range(3):
        num_anchors = tf.cast(len(anchor_mask[l]), dtype=tf.int32)
        grid_shape = tf.cast(grid_shapes[l], dtype=tf.int32)
        # (1, cel_row, 1, 1)
        row_idx = tf.cast(tf.reshape(tf.range(0, grid_shape[0]), [1, -1, 1, 1]), dtype=tf.int32)
        # (1, 1, cel_col, 1)
        col_idx = tf.cast(tf.reshape(tf.range(0, grid_shape[1]), [1, 1, -1, 1]), dtype=tf.int32)
        # (1, 1, 1, num_anchors)
        anc_idx = tf.cast(tf.reshape(tf.range(0, num_anchors), [1, 1, 1, -1]), dtype=tf.int32)
        # (bs, cel_row, cel_col, num_anchors)
        bs_row_col_anc = bs_idx * grid_shape[0] * grid_shape[1] * num_anchors + \
            row_idx * grid_shape[1] * num_anchors + col_idx * num_anchors + anc_idx

        # (bs, cel_row, cel_col, anchors_per_cel, 1)
        object_mask = y_true[l][..., 4:5]
        # (bs, cel_row, cel_col, anchors_per_cel, num_cls)
        true_class_probs = y_true[l][..., 5:5 + num_classes]

        # get YOLO output
        grid, raw_pred, pred_xy, pred_wh, pred_cs_conf, pred_cs, _ = \
            yolo_head(yolo_outputs[l],
                      anchors[anchor_mask[l]],
                      num_classes,
                      num_angle_bins,
                      input_shape, calc_loss=True)

        feature_xy_reshape, \
            feature_wh_reshape, \
            feature_conf_reshape, \
            feature_cls_reshape, \
            feature_hwl_reshape, \
            feature_cs_conf_reshape, \
            feature_cs_reshape = raw_pred

        # (bs, cel_row, cel_col, 3, 5)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]
        raw_true_wh = y_true[l][..., 2:4]

        # for 3D
        raw_true_cs_conf = \
            y_true[l][..., (5 + 4 * num_classes):(5 + 4 * num_classes + num_angle_bins)]
        # (bs, cel_row, cel_col, anchors_per_cel, 2*num_angle_bins)
        raw_true_cs = y_true[l][..., (5 + 4 * num_classes + num_angle_bins):]

        # grab the hwl
        # (bs, cel_row, cel_col, anchors_per_cel)
        c = tf.argmax(true_class_probs, axis=-1)
        # (bs, cel_row, cel_col, anchors_per_cel)
        flat_idx = tf.cast(bs_row_col_anc, dtype=tf.int32) * num_classes * 3 \
            + tf.cast(c, dtype=tf.int32) * 3
        flat_idx = tf.reshape(flat_idx, (bs, grid_shape[0], grid_shape[1], num_anchors, 1))
        flat_idx = tf.concat([flat_idx, flat_idx + 1, flat_idx + 2], axis=-1)
        # (bs, cel_row, cel_col, anchors_per_cel, 3)
        flat_idx = tf.reshape(flat_idx, [-1])
        # (bs*cel_row*cel_col*anchors_per_cel*3)
        y_true_flat = tf.reshape(y_true[l][..., 5 + num_classes:5 + 4 * num_classes], [-1])
        # (bs*cel_row*cel_col*anchors_per_cel*n)
        raw_true_hwl = tf.reshape(tf.gather(y_true_flat, flat_idx),
                                  [bs, grid_shape[0], grid_shape[1], num_anchors, 3])
        # NOTE: the pred_hwl from yolo_head is the predicted hwl according to predicted class,
        # but this class maybe wrong, so we have to get the predited hwl according to ground
        # truch class.
        pred_hwl = tf.reshape(feature_hwl_reshape, [-1])
        pred_hwl = tf.reshape(tf.gather(pred_hwl, flat_idx),
                              [bs, grid_shape[0], grid_shape[1], num_anchors, 3])
        pred_hwl = tf.exp(tf.clip_by_value(pred_hwl, -10, 5))

        # (bs, cel_row, cel_col, anchors_per_cel, 1)
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        # (bs, cel_row, cel_col, anchors_per_cel, 1)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            #(row, col, 3, 4), (row, col, 3) -> (?, 4)
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # (cel_row, cel_col, anchors_per_box, ?)
            iou = box_IoU(pred_box[b], true_box)
            # (cel_row, cel_col, anchors_per_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = \
            K.control_flow_ops.while_loop(lambda b, *args: b < bs, loop_body, [0, ignore_mask])
        # (bs, cel_row, cel_col, anchors_per_cel)
        ignore_mask = ignore_mask.stack()
        # (bs, cel_row, cel_col, anchors_per_cel, 1)
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # set some weights for different losses
        lambda_xy = cfg.lambda_xy
        lambda_wh = cfg.lambda_wh
        lambda_obj = cfg.lambda_obj
        lambda_nonobj = cfg.lambda_nonobj
        lambda_cls = cfg.lambda_cls
        lambda_cs_conf = cfg.lambda_cs_conf
        lambda_alpha = cfg.lambda_alpha
        lambda_hwl = cfg.lambda_hwl

        # (bs, cel_row, cel_col, anchors_per_cel, 2)
        xy_loss = lambda_xy * object_mask * \
            tf.square(raw_true_xy - pred_xy)

        # In YOLOv1, sum of square of sqrt loss is used,
        # but sqrt has nan problem in backprop, need some trick if you really want to use it
        # (bs, cel_row, cel_col, anchors_per_cel, 2)
        wh_loss = lambda_wh * object_mask * \
            tf.square(raw_true_wh - pred_wh)

        cs_conf_loss = lambda_cs_conf * object_mask * \
            tf.expand_dims(
                tf.nn.softmax_cross_entropy_with_logits(labels=raw_true_cs_conf,
                                                        logits=feature_cs_conf_reshape),
                -1)

        # cs means cos,sin
        # loss for alpha angle, simply use sum of square loss here,
        # you can replace by the arccos loss
        raw_true_cs = tf.reshape(raw_true_cs,
                                 [bs, grid_shape[0], grid_shape[1], num_anchors, num_angle_bins, 2])
        # to consider float value representation error
        cs_mask = tf.greater(tf.abs(raw_true_cs - 0.0), 0.00001)
        # (bs, grid_shape[0], grid_shape[1], num_anchors, num_angle_bins, 1)
        cs_mask = tf.logical_or(cs_mask[..., 0:1], cs_mask[..., 1:])
        # (bs, grid_shape[0], grid_shape[1], num_anchors, num_angle_bins*2)
        cs_mask = tf.cast(tf.reshape(tf.concat([cs_mask, cs_mask], axis=-1),
                                     [bs, grid_shape[0], grid_shape[1], num_anchors, -1]),
                          dtype=tf.float32)
        raw_true_cs = tf.reshape(raw_true_cs,
                                 [bs, grid_shape[0], grid_shape[1], num_anchors, -1])
        pred_cs = tf.nn.l2_normalize(
            tf.reshape(feature_cs_reshape,
                       [bs, grid_shape[0], grid_shape[1], num_anchors, num_angle_bins, 2]),
            dim=-1)
        pred_cs = tf.reshape(pred_cs, [bs, grid_shape[0], grid_shape[1], num_anchors, -1])
        # (bs, cel_row, cel_co, anchors_per_cell, 2*num_angle_bins)
        alpha_loss = lambda_alpha * object_mask * cs_mask * \
            tf.square(raw_true_cs - pred_cs)

        # (bs, cel_row, cel_col, anchors_per_cel, 3)
        hwl_loss = lambda_hwl * object_mask * \
            tf.square(raw_true_hwl - pred_hwl)

        positive_confidence_loss = lambda_obj * \
            object_mask * \
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=object_mask,
                logits=feature_conf_reshape)
        negative_confidence_loss = lambda_nonobj * \
            ignore_mask * \
            (1 - object_mask) * \
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=object_mask,
                logits=feature_conf_reshape)
        # (bs, cel_row, cel_col, anchors_per_cel, num_cls, 1)
        class_loss = lambda_cls * \
            object_mask * \
            tf.expand_dims(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=true_class_probs,
                    logits=feature_cls_reshape),
                -1)

        xy_loss = tf.reduce_sum(xy_loss) / tf.cast(bs, tf.float32)
        wh_loss = tf.reduce_sum(wh_loss) / tf.cast(bs, tf.float32)
        positive_confidence_loss = \
            tf.reduce_sum(positive_confidence_loss) / tf.cast(bs, tf.float32)
        negative_confidence_loss = \
            tf.reduce_sum(negative_confidence_loss) / tf.cast(bs, tf.float32)
        class_loss = tf.reduce_sum(class_loss) / tf.cast(bs, tf.float32)
        cs_conf_loss = tf.reduce_sum(cs_conf_loss) / tf.cast(bs, tf.float32)
        alpha_loss = tf.reduce_sum(alpha_loss) / tf.cast(bs, tf.float32)
        hwl_loss = tf.reduce_sum(hwl_loss) / tf.cast(bs, tf.float32)

        scale_loss = xy_loss + \
            wh_loss + \
            positive_confidence_loss + \
            negative_confidence_loss + \
            class_loss + \
            cs_conf_loss + \
            alpha_loss + \
            hwl_loss
        xy_loss_list.append(xy_loss)
        wh_loss_list.append(wh_loss)
        positive_conf_loss_list.append(positive_confidence_loss)
        negative_conf_loss_list.append(negative_confidence_loss)
        class_loss_list.append(class_loss)
        scale_loss_list.append(scale_loss)
        alpha_loss_list.append(alpha_loss)
        hwl_loss_list.append(hwl_loss)
    total_loss = tf.add_n(scale_loss_list)
    total_xy_loss = tf.add_n(xy_loss_list)
    total_wh_loss = tf.add_n(wh_loss_list)
    total_positive_conf_loss = tf.add_n(positive_conf_loss_list)
    total_negative_conf_loss = tf.add_n(negative_conf_loss_list)
    total_class_loss = tf.add_n(class_loss_list)
    total_alpha_loss = tf.add_n(alpha_loss_list)
    total_hwl_loss = tf.add_n(hwl_loss_list)
    return total_loss, total_xy_loss, total_wh_loss, \
        total_positive_conf_loss, total_negative_conf_loss, \
        total_class_loss, total_alpha_loss, total_hwl_loss


def box_IoU(b1, b2):
    """
    Calculer IoU between 2 BBs
    # hoi bi nguoc han tinh left bottom, right top TODO
    :param b1: predicted box, shape=[x, x, 3, 4], 4: xywh
    :param b2: true box, shape=[n, 4], 4: xywh
    :return: iou: intersection of 2 BBs, tensor, shape=[x, x, 3, n] ,1: IoU
    b = tf.cast(b, dtype=tf.float32)
    """
    with tf.name_scope('BB1'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        # shape= (13, 13, 3, 1, 4)
        b1 = tf.expand_dims(b1, -2)
        # x,y shape=(13, 13, 3, 1, 2)
        b1_xy = b1[..., :2]
        # w,h shape=(13, 13, 3, 1, 2)
        b1_wh = b1[..., 2:4]
        # w/2, h/2 shape= (13, 13, 3, 1, 2)
        b1_wh_half = b1_wh / 2.
        # x,y: left bottom corner of BB (13, 13, 3, 1, 2)
        b1_mins = b1_xy - b1_wh_half
        # x,y: right top corner of BB (13, 13, 3, 1, 2)
        b1_maxes = b1_xy + b1_wh_half
        # w1 * h1 (13, 13, 3, 1)
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]

    with tf.name_scope('BB2'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        # shape= (1, n, 4)
        b2 = tf.expand_dims(b2, 0)
        # x,y shape=(1, n, 2)
        b2_xy = b2[..., :2]
        # w,h shape=(1, n, 2)
        b2_wh = b2[..., 2:4]
        # w/2, h/2 shape=(1, n, 2)
        b2_wh_half = b2_wh / 2.
        # x,y: left bottom corner of BB (1, n, 2)
        b2_mins = b2_xy - b2_wh_half
        # x,y: right top corner of BB (1, n, 2)
        b2_maxes = b2_xy + b2_wh_half
        # w2 * h2 (1, n)
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    with tf.name_scope('Intersection'):
        """
        Calculate 2 corners: {left bottom, right top} 
        based on BB1, BB2 and area of this box
        """
        # (x,x,3,1,2), (1,n,2)->(x, x, 3, n, 2)
        intersect_mins = K.maximum(b1_mins, b2_mins)
        # (13, 13, 3, n, 2)
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        # (13, 13, 3, n, 2)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        # (13, 13, 3, n)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # (13, 13, 3, n)
    IoU = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-IoU')

    return IoU
