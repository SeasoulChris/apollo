#!/usr/bin/env python

import io
import math
import os
import sys
import time

from PIL import Image
from tensorflow.python.ops import control_flow_ops
import json
import numpy as np
import tensorflow as tf

from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.network.network_function import YOLOv3
from fueling.perception.YOLOv3.utils.loss_function import compute_loss
from fueling.perception.YOLOv3.utils.loss_function import convert_raw_output_to_box
from fueling.perception.YOLOv3.utils.object_utils import Object
from fueling.perception.YOLOv3.utils.projected_utils import kitti_obj_cam_interaction
from fueling.perception.YOLOv3.utils.projected_utils import read_camera_params
from fueling.perception.YOLOv3.utils.yolo_utils import accumulate_obj
from fueling.perception.YOLOv3.utils.yolo_utils import convert_to_original_size
from fueling.perception.YOLOv3.utils.yolo_utils import draw_boxes
from fueling.perception.YOLOv3.utils.yolo_utils import non_max_suppression
import fueling.common.logging as logging


GPU = cfg.gpu
BATCH_SIZE = cfg.batch_size
INPUT_HEIGHT = cfg.Input_height
INPUT_WIDTH = cfg.Input_width
CHANNELS = cfg.channels
NUM_ANCHOR_BOXES_PER_SCALE = cfg.num_anchor_boxes_per_scale
NUM_OUTPUT_LAYERS = cfg.num_output_layers
ANCHORS = cfg.anchors
NUM_CLASSES = cfg.num_classes
NUM_ANGLE_BINS = cfg.num_angle_bins
RESTORE_PATH = cfg.restore_path
NMS_CONFIDENCE_THRESHOLD = cfg.nms_confidence_threshold
NMS_IOU_THRESHOLD = cfg.nms_iou_threshold
CLASS_MAP = cfg.class_map
ORIGINAL_WIDTH = cfg.original_width
ORIGINAL_HEIGHT = cfg.original_height
MODEL_OUTPUT_PATH = cfg.model_output_path


os.environ["CUDA_VISIBLE_DEVICES"] = GPU
slim = tf.contrib.slim


class Inference:

    def __init__(self):
        pass

    def _init_essential_placeholders(self):
        """
        Essential placeholders.
        """
        placeholders = {}
        placeholders["input_image"] = tf.placeholder(tf.float32,
                                                     shape=[BATCH_SIZE, INPUT_HEIGHT,
                                                            INPUT_WIDTH, CHANNELS],
                                                     name="Input")
        return placeholders

    def _config_graph(self, input_tensor, is_training=True, reuse=False):
        """
        Configure the inference computation graph.
        """
        scale1, scale2, scale3, feature1, feature2, feature3 = \
            YOLOv3(input_tensor, ANCHORS, NUM_CLASSES, is_training)\
            .yolo_v3(num_layers=NUM_OUTPUT_LAYERS,
                     num_anchors_per_cell=NUM_ANCHOR_BOXES_PER_SCALE,
                     reuse=reuse)
        return scale1, scale2, scale3, feature1, feature2, feature3

    def _restore_from_checkpoint(self, sess):
        """
        Restore training from a checkpoint.
        """
        restore_saver = tf.train.Saver()
        restore_saver.restore(sess, RESTORE_PATH)
        logging.info("Restored weights from {}.".format(RESTORE_PATH))

    def setup_network(self):
        """
        Start training.
        """
        self.essential_placeholders = self._init_essential_placeholders()
        output_scale1, output_scale2, output_scale3, feature1, feature2, feature3 \
            = self._config_graph(self.essential_placeholders["input_image"], is_training=False)

        self.xy_wh_conf = \
            convert_raw_output_to_box([output_scale1, output_scale2, output_scale3],
                                      ANCHORS)

        # start session and train
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._restore_from_checkpoint(self.sess)

    def run(self, data, output_dir):
        """
        Perform 1 update on the model with input training data.
        """
        image_batch, _, _, _, cls_box_map_lists, objs_list, \
            calib_list, image_name_list, original_image_list = data
        feed_dict = {
            self.essential_placeholders["input_image"]: (image_batch / 255.)}

        xy_wh_conf_value = self.sess.run(self.xy_wh_conf, feed_dict=feed_dict)
        detection_string_list_batch, boxes = accumulate_obj(xy_wh_conf_value,
                                                            calib_batch=calib_list)

        def _write_output():
            for batch_id, image_dets in enumerate(detection_string_list_batch):
                with open(os.path.join(output_dir,
                             "{}.txt".format(image_name_list[batch_id])), "w") as handle:
                    for line_id, line in enumerate(image_dets):
                        if (line_id != len(image_dets) - 1):
                            line = "{}\n".format(line)
                        handle.write(line)
        def _write_image():
            cls_names = {v:k for k, v in CLASS_MAP.items()}
            for i in range(len(boxes)):
                draw_boxes(boxes[i], original_image_list[i],
                           cls_names,
                           (INPUT_WIDTH, INPUT_HEIGHT),
                           (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                           calib_list[i], False,
                           cls_box_map=cls_box_map_lists[i])
                original_image_list[i].save(os.path.join(output_dir,
                    "{}.jpg".format(image_name_list[i])))
        _write_output()
        _write_image()
