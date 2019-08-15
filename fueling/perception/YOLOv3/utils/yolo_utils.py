#!/usr/bin/env python

import collections
import math
import os
import random

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from fueling.perception.YOLOv3.utils.data_augmentation import *
from fueling.perception.YOLOv3.utils.object_utils import Object, Label_Object, Calibration
from fueling.perception.YOLOv3.utils.projected_utils import read_camera_params
from fueling.perception.YOLOv3.utils.projected_utils import kitti_obj_cam_interaction
from fueling.perception.YOLOv3.utils.projected_utils import draw_3d_box
import fueling.perception.YOLOv3.config as cfg


def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size):
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter boxed.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos

def convert_to_original_size(box, size, original_size, is_letter_box_image):
    if is_letter_box_image:
        box = box.reshape(2, 2)
        box[0, :] = letter_box_pos_to_original_pos(box[0, :], size, original_size)
        box[1, :] = letter_box_pos_to_original_pos(box[1, :], size, original_size)
    else:
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))

def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1-int_x0, 0) * max(int_y1-int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def non_max_suppression(predictions_with_boxes, confidence_threshold=0.9, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array,
              [bs, ...+...+..., 10:(xmin, ymin, xmax, ymax, cos, sin, h, w, l)]
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1) # (bs, ...+...+..., 1)
    # The line below "non_zero_idxs = np.nonzero(image_pred)" assums predictions are all non-zero,
    # so add 1e-10 here
    predictions = (predictions_with_boxes + 1e-10) * conf_mask # (bs, ...+...+..., 5+num_cls)

    results = []
    for i, image_pred in enumerate(predictions):
        result = {}
        shape = image_pred.shape # [x, 5+num_cls]
        non_zero_idxs = np.nonzero(image_pred) 
        image_pred = image_pred[non_zero_idxs] # (...)
        image_pred = image_pred.reshape(-1, shape[-1]) # (x, 5+num_cls)

        bbox_attrs = image_pred[:, :10]
        classes = image_pred[:, 10:]
        classes = np.argmax(classes, axis=-1) # (x, )

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls # (x, )
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, 4].argsort()[::-1]]
            cls_scores = cls_boxes[:, 4]
            cls_cshwl = cls_boxes[:, 5:10]
            cls_boxes = cls_boxes[:, 0:4]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                cshwl = cls_cshwl[0]
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score, cshwl))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                cls_cshwl = cls_cshwl[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
                cls_cshwl = cls_cshwl[np.nonzero(iou_mask)]
        results.append(result)

    return results

def draw_boxes(boxes, img, cls_names, detection_size,
               orig_size, calib, is_letter_box_image, cls_box_map=None):
    """
    boxes: dictionary cls_id -> list of objests. objects: (2d bbox, confidence, cshwl);
             2d bbox:xmin,ymin,xmax,ymax; cshwl: cos(local_angle), sin(local_angle), h, w, l.
    img: PIL img to plot boxes on
    cls_names: dict cls_id -> class name. 
    detection_size: network input image size. (width, height)
    orig_size: original image size. (width, height)
    calib: a KITTI Calibration object 
    is_letter_box_image: 
    cls_box_map: dict cls_id->list of 2d bbox of ground truth objects.
    """
    draw = ImageDraw.Draw(img)
    interactor = kitti_obj_cam_interaction(calib)

    original_width, original_height = orig_size
    objs = []
    for cls, bboxs in boxes.items():
        color = (0, 0, 255)
        #font = ImageFont.truetype("arial.ttf", 8)
        for box_o, score, cshwl in bboxs:
            box = convert_to_original_size(box_o, np.array(detection_size),
                                           np.array(img.size),
                                           is_letter_box_image)
            if len(bboxs) < 20:
                obj = Object([cls_names[cls], None, None, None, 
                              box_o[0]*original_width/detection_size[0], 
                              box_o[1]*original_height/detection_size[1], 
                              box_o[2]*original_width/detection_size[0], 
                              box_o[3]*original_height/detection_size[1], 
                              cshwl[2], cshwl[3], cshwl[4],
                              None, None, None, None])
                obj.score = score
                local_angle = math.degrees(np.arctan2(cshwl[1], cshwl[0]))
                beta = interactor.local_angle_to_car_yaw(local_angle, obj)
                obj.ry = math.radians(beta)
                translation = interactor.compute_translation(obj)
                if translation != None:
                    obj.t = translation
                    points_cam = \
                        interactor.bbox_from_local_angle_translation_dimension(obj, local_angle)
                    # image_points shape: (2, n)
                    image_points = \
                        interactor.project_to_image(points_cam, point_in_ref_cam=False)
                    ratio = np.array(img.size) / np.array([original_width, original_height])
                    image_points = image_points.transpose() * ratio #(n, 2)
                    draw_3d_box(img, image_points)
                    #obj.t = tuple(np.array(translation) - interactor.offset.reshape((3,)) )
                objs.append(obj) 
       
            draw.rectangle(box, outline=color)
            #draw.text(box[:2], '{} {:.2f}%'.format(
            #    cls_names[cls], score * 100), fill=color, font=font)
    
    if cls_box_map:
        color = (255, 255, 0)
        for cls, bboxs in cls_box_map.items():
            for box in bboxs:
                box = convert_to_original_size(box, np.array(detection_size),
                                               np.array(img.size),
                                               is_letter_box_image)
                draw.rectangle(box, outline=color)
   
    return objs

def process_label_file(file_path, image_dir, calib_dir, input_shape, anchors,
                       cls_name_id_map, num_classes, num_angle_bins,
                       bin_overlap_frac, num_output_layers,
                       random_color_shift=False, color_shift_percentage=0.5,
                       random_crop_=False, random_flip=False,
                       flip_chance=0.5,  random_jitter_=False,
                       jitter_chance=0.5, jitter_percentage=0.1,  max_boxes=100):
    """
    params:
    file_path: path to txt label file. Each line in the file should be a bbox->
               (class_name, x*3, x_min, y_min, x_max, y_max, x*8, class_id, x*8)
               [0:truncated, 1:occluded, 2:alpha, 3:x0, 4:y0, 5:x1, 6:y1, 
                7:h, 8:w, 9:l, 10:X, 11:Y, 12:Z, 13:rotation_y]
    image_dir: path to directory that contains all the images. The label txt file name 
               should be the same with image names inside image_dir.
    input_shape: network input shape -> (input_height, input_width)
    anchors: array of shape (9 anchors, 2->w,h)

    return:
    image_data: image in the [0, 255] uint8 scale
    y_true: [3 scales * (N->1, cell_rows, cell_cols, anchors_per_cell, 10+num_class)]
              (0:4)-> xywh normalized w.r.t input width and height, objectness,
                      cos(alpha), sin(alpha), 3d_h, 3d_w, 3d_l, + probability_of_classes
    """
    image_jpg_path = os.path.join(image_dir, os.path.basename(file_path).split('.')[0]+".jpg")
    image_png_path = os.path.join(image_dir, os.path.basename(file_path).split('.')[0]+".png")
    if not os.path.exists(file_path):
        raise RuntimeError("Label file path : {} does not exist.".format(file_path))
    if os.path.exists(image_jpg_path):
        image_path = image_jpg_path
    elif os.path.exists(image_png_path):
        image_path = image_png_path
    else:
        raise RuntimeError("Image file path : {}/png does not exist.".format(image_jpg_path)) 
    image_temp = Image.open(image_path)
    image = image_temp.copy()
    origin_image_size = image.size
    image_temp.close()   

    calib_path = os.path.join(calib_dir, os.path.basename(file_path).split('.')[0]+".txt")
    if not os.path.exists(calib_path):
        raise RuntimeError("Calibration file path : {} does not exist.".format(calib_path))
    calib = read_camera_params(calib_path)
    interactor = kitti_obj_cam_interaction(calib)

    image_np = np.asarray(image, dtype=np.uint8)
    start_x, start_y, crop_width, crop_height = None, None, None, None
    flip_switch = random_flip and random.uniform(0.0, 1.0) < flip_chance
    if random_color_shift:
        image_np = random_hsv_shift(image_np, low=1-color_shift_percentage,
                                    high=1+color_shift_percentage)
    if flip_switch:
        image_np = flip_image_lr(image_np)
    if random_crop_:
        image_np, start_x, start_y, crop_width, crop_height = random_crop(image_np)
    image = Image.fromarray(image_np)

    # resize the image to input shape
    resized_image, _ = letterbox_image(image, (input_shape[1], input_shape[0]))

    image_data = np.array(resized_image, dtype=np.uint8)  # pixel: [0:255]
    box_data = []
    # 9 -> x_min, y_min, x_max, y_max, class_id, alpha, 3d_h, 3d_w, 3d_l
    boxes = np.zeros((max_boxes, 9), dtype=np.float)
    input_shape = np.array(input_shape)[::-1] #  hw -> wh 
    image_size = np.array(image.size)
    with open(file_path) as handle:
        objs = [Label_Object(line, cls_name_id_map)
                for line in handle.readlines()
                if line.split(' ')[0] in cls_name_id_map.keys()]
        # TODO[Kawai]: remove following to include truncation
        objs = [obj for obj in objs if (obj.truncation<=0.0001 and obj.occlusion<5)]
    # correct the BBs to the image resize
    cls_box_map = collections.defaultdict(list)
    if len(objs)==0:  # if there is no object in this image
        box_data.append(boxes)
    else:
        i = 0
        for obj in objs:
            #if flip_switch:
            #    new_box = flip_bbox_lr(box[0][3:7], origin_image_size[0])
            #else:
            #    new_box = box[0][3:7]

            #if random_crop_ and new_box!=None:
            #    new_box = random_crop_bbox(new_box, start_x, start_y, crop_width, crop_height)

            #if random_jitter_ and new_box!=None and random.uniform(0.0, 1.0) < jitter_chance:
            #    new_box = random_jitter(new_box, jitter=jitter_percentage)

            if i < max_boxes: 
                #if new_box != None:
                #    boxes[i, 0:5] = np.array(new_box + [box[1]])
                #    boxes[i, 0] = max(0, min(image_size[0]-1, boxes[i, 0]))
                #    boxes[i, 2] = max(0, min(image_size[0]-1, boxes[i, 2]))
                #    boxes[i, 1] = max(0, min(image_size[1]-1, boxes[i, 1]))
                #    boxes[i, 3] = max(0, min(image_size[1]-1, boxes[i, 3]))
                #else:
                #    continue
                boxes[i, 0:5] = np.array([obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.type_id])
                boxes[i, 0] = max(0, min(image_size[0]-1, boxes[i, 0]))
                boxes[i, 2] = max(0, min(image_size[0]-1, boxes[i, 2]))
                boxes[i, 1] = max(0, min(image_size[1]-1, boxes[i, 1]))
                boxes[i, 3] = max(0, min(image_size[1]-1, boxes[i, 3]))
                boxes[i, 5] = math.radians(interactor.angle_btw_car_and_2d_bbox(obj))
                boxes[i, 6:9] = np.array([obj.h, obj.w, obj.l])
            else:
                break

            # Convert 2d boxes to network input shape scale
            boxes[i, 0] = (boxes[i, 0]*input_shape[0]/image_size[0]).astype(np.int32)
            boxes[i, 2] = (boxes[i, 2]*input_shape[0]/image_size[0]).astype(np.int32)
            boxes[i, 1] = (boxes[i, 1]*input_shape[1]/image_size[1]).astype(np.int32)
            boxes[i, 3] = (boxes[i, 3]*input_shape[1]/image_size[1]).astype(np.int32)
            assert (np.array(boxes)[i, 0] < input_shape[0]).all()
            assert (np.array(boxes)[i, 2] < input_shape[0]).all()
            assert (np.array(boxes)[i, 1] < input_shape[1]).all()
            assert (np.array(boxes)[i, 3] < input_shape[1]).all()
            # map class ids to a list of all objs of that class
            cls_box_map[obj.type_id].append(np.array(boxes[i, :4]))
            i += 1
        box_data.append(boxes)
    # convet to YOLO output format
    y_true = preprocess_true_boxes(np.array(box_data), input_shape, anchors,
                                   num_classes, num_angle_bins,
                                   bin_overlap_frac, num_output_layers)
    return image_data, y_true, cls_box_map, objs, calib

def letterbox_image(image, size):
    """resize image, not keeping the aspect ratio. No padding
    :param: size: (input_width, input_height)
    :return:
    boxed_image: the resized image
    image_shape: original image shape (h, w)
    """
    image_w, image_h = image.size
    image_shape = np.array([image_h, image_w])
    w, h = size
    resized_image = image.resize((w, h), Image.BICUBIC)
    
    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image)

    return boxed_image, image_shape

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes,
                          num_angle_bins, bin_overlap_frac, num_output_layers):
    """
    Preprocess true boxes to training input format
    :param true_boxes: array, shape=(bs, max_num_boxes, 9) 
                       9-> (0:x_min, 1:y_min, 2:x_max, 3:y_max, 4:class_id, 5:ry, 6:h, 7:w, 8:l)
    :param input_shape: network input shape, array, wh, multiples of 32, shape = (2,)
    :param anchors: array, shape=(9 arches per cell, 2 -> w,h)
    :param num_classes: integer
    :param num_angle_bins: integer, the number of alpha angle bins to partition the angle into
    :return: y_true: [3 scales * (N, cell_rows, cell_cols, anchors_per_cell, 5+num_class*4+2)]
                       (0:4)-> xywh normalized w.r.t input width and height, objectness,
                       (5:5+num_class):class_prob,
                       (5+num_cls:5+num_cls*4):class_spec_hwl,
                       (5+num_cls*4:5+num_cls*4+num_angle_bins):algle_bin_id,
                       (5+num_cls*4+num_angle_bins:5+num_cls*4+num_angle_bins*3):
                         cos(alpha), sin(alpha)
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    anchor_mask = cfg.anchor_mask # different anchors are assigned to different scales (3, 3)
    # (bs, max_box, 9)  0:4 -> (x_min, y_min, x_max, y_max)
    true_boxes = np.array(true_boxes, dtype=np.float32)

    # (x, y) center point of BB (bs, max_box, 2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    # w = x_max - x_min  (bs, max_box, 2)
    # h = y_max - y_min
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    # (bs, max_box, 9) 0:4 -> (x, y, w, h) normalized
    true_boxes[..., 0:2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh / input_shape

    BS = true_boxes.shape[0]
    # (3 scales, 2 ->x,y)
    grid_shapes = [input_shape // scale for scale in (32, 16, 8)]
    # [3 x (BS, grid_shapes[i][0],  grid_shapes[i][1], 3, 5+num_class*4+2)]
    y_true = [np.zeros((BS, grid_shapes[l][1], grid_shapes[l][0],
                        len(anchor_mask[l]), num_output_layers),
                       dtype=np.float32) for l in range(len(grid_shapes))]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # (1, num_anchors, 2)
    anchor_maxes = anchors / 2.  # w/2, h/2  (1, num_anchors, 2)
    anchor_mins = -anchor_maxes   # -w/2, -h/2  (1, num_anchors, 2)
    valid_mask = boxes_wh[..., 0] > 0  # w>0 True, w=0 False (BS, max_box)

    # Compute angle ranges base on num_angle_bis
    principle_angle = (2 * np.pi / num_angle_bins)
    if num_angle_bins == 1:
        lower_bound = np.array([0.0])
        upper_bound = np.array([2 * np.pi])
    else:
        lower_bound = np.array([x*principle_angle for x in range(num_angle_bins)])
        upper_bound = lower_bound + principle_angle
        lower_bound = lower_bound - principle_angle * bin_overlap_frac
        upper_bound = upper_bound + principle_angle * bin_overlap_frac
        lower_bound[0] = lower_bound[0] + (2 * np.pi)
        upper_bound[-1] = upper_bound[-1] - (2 * np.pi)

    for b in (range(BS)):  # for all of BS image
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  # (?, 2)
        valid_objs = true_boxes[b, valid_mask[b]] # (?, 9)
        # Expand dim to apply broadcasting.
        if len(wh)==0:
            continue
        wh = np.expand_dims(wh, -2) # (?, 1, 2)
        box_maxes = wh / 2.         # (?, 1, 2)
        box_mins = -box_maxes       # (?, 1, 2)

        # (?, num_anchors, 2) -> (x_min, y_min)
        intersect_mins = np.maximum(box_mins, anchor_mins)
        # (?, num_anchors, 2) -> (x_max, y_max)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        # (?, num_anchors, 2) -> (intersec_w, intersec_h)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        # (?, num_anchors) -> intersec_area
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        # (?, num_anchors) -> iou
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        # (?, )
        best_anchor = np.argmax(iou, axis=-1)
        
        for t, n in enumerate(best_anchor): # t, also the t-th obj in valid_objs
            for l in range(len(grid_shapes)):  # 1 in 3 scale
                # choose the corresponding mask: best_anchor in [0, 1, 2]or[3, 4, 5]or[6, 7, 8]
                if n in anchor_mask[l]:
                    # j: row of cell
                    j = np.floor(valid_objs[t, 1] * grid_shapes[l][1]).astype(np.int32)
                    # i: col or cell
                    i = np.floor(valid_objs[t, 0] * grid_shapes[l][0]).astype(np.int32)

                    # k: which anchor in l-th scale
                    k = anchor_mask[l].index(n)
                    # c: class index
                    c = valid_objs[t, 4].astype(np.int32)
                    # l: scale; b; idx image; grid(i:col , j:row);
                    # k: which anchor; 0:4: (x,y,w,h)/input_shape
                    y_true[l][b, j, i, k, 0:4] = valid_objs[t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1.0  # score = 1
                    # for 3D
                    y_true[l][b, j, i, k, 5 + c] = 1.0 
                    y_true[l][b, j, i, k, 5+int(num_classes)+3*c:5+int(num_classes)+3*c+3] = \
                        valid_objs[t, 6:9] 

                    # angles
                    if valid_objs[t, 5] < 0.0:
                        valid_objs[t, 5] = valid_objs[t, 5] + (2 * np.pi)
                    lower_mask = lower_bound < valid_objs[t, 5]
                    upper_mask = upper_bound >= valid_objs[t, 5]
                    angle_mask = np.logical_and(lower_mask, upper_mask)
                    angle_mask[0] = np.logical_or(lower_mask[0], upper_mask[0])
                    angle_mask[-1] = np.logical_or(lower_mask[-1], upper_mask[-1])
                    assert (np.sum(angle_mask)<=2,
                               "obj angles should not fall into more than 2 angle bins.")
                    bins = np.where(angle_mask==True)[0]
                    regression_angles = np.array([valid_objs[t, 5] - \
                                                  (idx*principle_angle+(principle_angle/2))
                                                  for idx in bins])
                    min_idx = np.argmin([abs(ang) for ang in regression_angles])
                    primary_bin = bins[min_idx]
                    
                    y_true[l][b, j, i, k, 5+4*int(num_classes) + primary_bin] = 1.0
                    for bin_idx, reg_ang in zip(bins, regression_angles):
                        y_true[l][b, j, i, k,
                                  5+4*int(num_classes)+int(num_angle_bins)+2*bin_idx] = \
                            np.cos(reg_ang)
                        y_true[l][b, j, i, k,
                                  5+4*int(num_classes)+int(num_angle_bins)+2*bin_idx+1] = \
                            np.sin(reg_ang)
                    break
    return y_true
