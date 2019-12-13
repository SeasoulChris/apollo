#!/usr/bin/env python

import io
import math
import os
import resource
import sys
import time

from PIL import Image
from tensorflow.python.client import device_lib
from tensorflow.python.ops import control_flow_ops
import json
import numpy as np
import tensorflow as tf

from fueling.perception.YOLOv3 import config as cfg
from fueling.perception.YOLOv3.network.network_function import YOLOv3
from fueling.perception.YOLOv3.utils.loss_function import compute_loss
from fueling.perception.YOLOv3.utils.loss_function import convert_raw_output_to_box
from fueling.perception.YOLOv3.utils.object_utils import Object
from fueling.perception.YOLOv3.utils.projected_utils import read_camera_params
from fueling.perception.YOLOv3.utils.projected_utils import kitti_obj_cam_interaction
from fueling.perception.YOLOv3.utils.yolo_utils import non_max_suppression
from fueling.perception.YOLOv3.utils.yolo_utils import draw_boxes
from fueling.perception.YOLOv3.utils.yolo_utils import convert_to_original_size
import fueling.common.logging as logging
import fueling.perception.YOLOv3.utils.data_utils as data_utils


GPU = cfg.gpu
CLASSES_TO_CONSIDER = cfg.classes_to_consider
BATCH_SIZE = cfg.batch_size
INPUT_HEIGHT = cfg.Input_height
INPUT_WIDTH = cfg.Input_width
CHANNELS = cfg.channels
VISUAL_SCALE = cfg.visual_scale
NUM_ANCHOR_BOXES_PER_SCALE = cfg.num_anchor_boxes_per_scale
NUM_OUTPUT_LAYERS = cfg.num_output_layers
WEIGHT_DECAY = cfg.weight_decay
ANCHORS = cfg.anchors
NUM_CLASSES = cfg.num_classes
NUM_ANGLE_BINS = cfg.num_angle_bins
NEGATIVE_IGNORE_THRESH = cfg.negative_ignore_thresh
START_FROM_COCO = cfg.start_from_coco
NMS_CONFIDENCE_THRESHOLD = cfg.nms_confidence_threshold
NMS_IOU_THRESHOLD = cfg.nms_iou_threshold
CLASS_MAP = cfg.class_map
ORIGINAL_WIDTH = cfg.original_width
ORIGINAL_HEIGHT = cfg.original_height
TRAIN_ONLY_VARIABLES = cfg.train_only_variables
LEARNING_RATE = cfg.learning_rate
DECAY_STEPS = cfg.decay_steps
DECAY_RATE = cfg.decay_rate
MAX_ITER = cfg.max_iter
SUMMARY_INTERVAL = cfg.summary_interval
PRINT_INTERVAL = cfg.print_interval
SAVE_INTERVAL = cfg.save_interval
MODEL_NAME_PREFIX = cfg.model_name_prefix

slim = tf.contrib.slim


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def print_current_memory_usage(step_name):
    mb_2_kb = 1024
    meminfo = dict((m.split()[0].rstrip(':'), int(m.split()[1]))
                   for m in open('/proc/meminfo').readlines())
    total_mem = meminfo['MemTotal'] // mb_2_kb
    used_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // mb_2_kb
    logging.info(f'step: {step_name}, total memory: {total_mem} MB, current memory: {used_mem} MB')


class training:

    def __init__(self, restore_path, start_step, num_gpu=1):
        self.model_restore_path = restore_path
        self.last_save_step = -1
        self.num_gpu = num_gpu
        self.epoch = 0
        self.cur_step = start_step + data_utils.get_latest_step(restore_path, MODEL_NAME_PREFIX)
        self.global_step = tf.Variable(self.cur_step, name='global_step', trainable=False)

    def _init_essential_placeholders(self):
        """
        Essential placeholders.
        """
        placeholders = {}
        placeholders["input_image"] = tf.compat.v1.placeholder(tf.float32,
                                                               shape=[BATCH_SIZE, INPUT_HEIGHT,
                                                                      INPUT_WIDTH, CHANNELS],
                                                               name="Input")
        placeholders["visual_image"] = tf.compat.v1.placeholder(tf.float32,
                                                                shape=[1, INPUT_HEIGHT * VISUAL_SCALE,
                                                                       INPUT_WIDTH * VISUAL_SCALE, CHANNELS],
                                                                name="visual_image")
        placeholders["is_train_placeholder"] = tf.compat.v1.placeholder(tf.bool, shape=[])
        with tf.name_scope("Target"):

            placeholders["label_scale1"] = \
                tf.compat.v1.placeholder(tf.float32,
                                         shape=[BATCH_SIZE, INPUT_HEIGHT / 32, INPUT_WIDTH / 32,
                                                NUM_ANCHOR_BOXES_PER_SCALE, (NUM_OUTPUT_LAYERS)],
                                         name="target_S1")
            placeholders["label_scale2"] = \
                tf.compat.v1.placeholder(tf.float32,
                                         shape=[BATCH_SIZE, INPUT_HEIGHT / 16, INPUT_WIDTH / 16,
                                                NUM_ANCHOR_BOXES_PER_SCALE, (NUM_OUTPUT_LAYERS)],
                                         name="target_S2")
            placeholders["label_scale3"] = \
                tf.compat.v1.placeholder(tf.float32,
                                         shape=[BATCH_SIZE, INPUT_HEIGHT / 8, INPUT_WIDTH / 8,
                                                NUM_ANCHOR_BOXES_PER_SCALE, (NUM_OUTPUT_LAYERS)],
                                         name="target_S3")
        return placeholders

    def _config_graph(self, input_tensor, is_training=True, reuse=False):
        """
        Configure the inference computation graph.
        """

        scale1, scale2, scale3, feature1, feature2, feature3 = \
            YOLOv3(input_tensor, ANCHORS, NUM_CLASSES, is_training)\
            .yolo_v3(num_layers=NUM_OUTPUT_LAYERS,
                     num_anchors_per_cell=NUM_ANCHOR_BOXES_PER_SCALE,
                     weight_decay=WEIGHT_DECAY,
                     reuse=reuse)
        return scale1, scale2, scale3, feature1, feature2, feature3

    def _compute_loss(self, gt_placeholder, scale1, scale2, scale3):
        """
        compute the loss.
        """
        with tf.name_scope("Loss_and_Detect"):
            # Label
            y_gt = [gt_placeholder["label_scale1"],
                    gt_placeholder["label_scale2"],
                    gt_placeholder["label_scale3"]]
            # Calculate loss
            loss = compute_loss([scale1, scale2, scale3],
                                y_gt,
                                ANCHORS,
                                NUM_CLASSES,
                                NUM_ANGLE_BINS,
                                ignore_thresh=NEGATIVE_IGNORE_THRESH,
                                print_loss=False)
        return loss

    def _init_optimizer(self, loss, start_learning_rate,
                        decay_steps, decay_rate, variable_list=None):
        """
        Initialize an optimizer.
        """
        learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=start_learning_rate,
                                                             global_step=self.global_step,
                                                             decay_steps=decay_steps,
                                                             decay_rate=decay_rate,
                                                             staircase=True)
        with tf.name_scope("Optimizer"):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss, variable_list,
                                                colocate_gradients_with_ops=True)
        return grads, optimizer

    def _init_model_saver(self):
        """
        Initialize a saver to save/restore model
        """
        saver = tf.compat.v1.train.Saver(max_to_keep=None)
        return saver

    def _restore_from_checkpoint(self, sess):
        """
        Restore training from a checkpoint.
        """
        if START_FROM_COCO:
            variables = [v for v in tf.compat.v1.global_variables() if ("Adam" not in v.name and
                                                                        "global_step" not in v.name and
                                                                        "Optimizer" not in v.name and
                                                                        "yolo-v3/Conv_14" not in v.name and
                                                                        "yolo-v3/Conv_22" not in v.name and
                                                                        "yolo-v3/Conv_6" not in v.name and
                                                                        "beta1_power" not in v.name and "beta2_power" not in v.name)]
            restore_saver = tf.train.Saver(var_list=variables)
        else:
            variables = [v for v in tf.compat.v1.global_variables() if ("Adam" not in v.name)]
            restore_saver = tf.compat.v1.train.Saver(var_list=variables)

        # TODO(longtao): figure why restore need a file name but not checkpoint folder
        restore_file_path = data_utils.get_restore_file_path(self.model_restore_path,
                                                             cfg.model_name_prefix)
        if restore_file_path:
            restore_saver.restore(sess, restore_file_path)
            logging.info(f'Restored weights from {restore_file_path}.')

    def _image_summary(self, image_batch, xy_wh_conf_value, calib_list, cls_box_map=None):
        """
        Add image summray.
        """
        boxes = non_max_suppression(xy_wh_conf_value,
                                    [INPUT_WIDTH, INPUT_HEIGHT],
                                    confidence_threshold=NMS_CONFIDENCE_THRESHOLD,
                                    iou_threshold=NMS_IOU_THRESHOLD)

        cls_names = {v: k for k, v in CLASS_MAP.items()}
        images = []
        for i in range(image_batch.shape[0]):
            img = Image.fromarray(np.uint8(image_batch[i, ...]))
            draw_boxes(boxes[i], img, cls_names,
                       (INPUT_WIDTH, INPUT_HEIGHT),
                       (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                       calib_list[i],
                       False, cls_box_map=cls_box_map[i] if cls_box_map else None)
            img = img.resize((INPUT_WIDTH * VISUAL_SCALE,
                              INPUT_HEIGHT * VISUAL_SCALE),
                             Image.BILINEAR)
            images.append(np.uint8(np.array(img)))
        return images

    def _init_summary_writer(self, suffix):
        """
        Initialize a summary writer.
        """
        return tf.compat.v1.summary.FileWriter(os.path.join(self.model_restore_path, suffix))

    def _add_to_summary(self, tensor, name, _type="scalar"):
        """
        Write tensor to summary.
        """
        if _type == "scalar":
            return tf.compat.v1.summary.scalar(name=name, tensor=tensor)
        elif _type == "image":
            return tf.compat.v1.summary.image(name=name, tensor=tensor)
        else:
            raise RuntimeError("Currently only support scaler and image.")

    def _add_all_scalar_summary(self, tensor_name_map):
        """
        Add all the tensors in tensor_nam_map into summary as scalars.
        """
        tensor_list = []
        for t, n in tensor_name_map.items():
            tensor_list.append(self._add_to_summary(t, n))
        return tf.compat.v1.summary.merge(tensor_list)

    def _add_all_image_summary(self, name_tensor_map):
        """
        Add all the tensors in tensor_name_map into summary as images.
        """
        tensor_list = []
        for n, t in name_tensor_map.items():
            tensor_list.append(self._add_to_summary(t, n, _type="image"))
        return tf.compat.v1.summary.merge(tensor_list)

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def setup_training(self):
        """
        Start training.
        """
        with tf.device("/cpu:0"):
            # Set random seed
            tf.compat.v1.set_random_seed(2)

            self.gpu_placeholders = []
            tower_grads = []
            reuse = False
            for i in range(self.num_gpu):
                with tf.device("/gpu:%d" % i):
                    e_placeholders = self._init_essential_placeholders()
                    if i == self.num_gpu - 1:
                        output_scale1, output_scale2, output_scale3, feature1, feature2, feature3 \
                            = tf.cond(e_placeholders["is_train_placeholder"],
                                      true_fn=lambda: self._config_graph(e_placeholders["input_image"],
                                                                         is_training=True,
                                                                         reuse=reuse),
                                      false_fn=lambda: self._config_graph(e_placeholders["input_image"],
                                                                          is_training=False,
                                                                          reuse=True))
                    else:
                        output_scale1, output_scale2, output_scale3, feature1, feature2, feature3 \
                            = self._config_graph(e_placeholders["input_image"],
                                                 is_training=True, reuse=reuse)
                    reuse = True

                    loss, xy_loss, wh_loss, positive_conf_loss, \
                        negative_conf_loss, cls_loss, alpha_loss, \
                        hwl_loss = self._compute_loss(e_placeholders, output_scale1,
                                                      output_scale2, output_scale3)
                    regularization_loss = tf.add_n(slim.losses.get_regularization_losses())

                    variables_to_train = None
                    if TRAIN_ONLY_VARIABLES:
                        variables_to_train = set()
                        for n in TRAIN_ONLY_VARIABLES:
                            temp = [v for v in tf.compat.v1.global_variables() if (n in v.name)]
                            variables_to_train = variables_to_train.union(set(temp))
                        variables_to_train = list(variables_to_train)
                        logging.info(f'=======Number of variables to train : {len(variables_to_train)}========')

                    grads, optimizer = self._init_optimizer(loss + regularization_loss,
                                                            start_learning_rate=LEARNING_RATE,
                                                            decay_steps=DECAY_STEPS,
                                                            decay_rate=DECAY_RATE,
                                                            variable_list=variables_to_train)
                    self.xy_wh_conf = \
                        convert_raw_output_to_box([output_scale1, output_scale2, output_scale3],
                                                  ANCHORS)
                    tower_grads.append(grads)
                    self.gpu_placeholders.append(e_placeholders)
            average_grads = self._average_gradients(tower_grads)

            # TODO[KaWai]: this is just an approximation of the real batch norm across multiple GPU.
            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(average_grads)

            self.saver = self._init_model_saver()

            # ====================All summaries ====================
            self.summary_writer_train = self._init_summary_writer(
                "summary")

            summary_tensor_map_train = {loss: "TRAIN_total_loss",
                                        xy_loss: "TRAIN_xy_loss",
                                        wh_loss: "TRAIN_wh_loss",
                                        positive_conf_loss: "TRAIN_positive_confidence_loss",
                                        negative_conf_loss: "TRAIN_negative_confidence_loss",
                                        cls_loss: "TRAIN_class_loss",
                                        alpha_loss: "TRAIN_alpha_loss",
                                        hwl_loss: "TRAIN_hwl_loss"}
            self.summary_op_train = self._add_all_scalar_summary(summary_tensor_map_train)
            self.image_summary_op = self._add_to_summary(e_placeholders["visual_image"],
                                                         "TRAIN_visualization",
                                                         _type="image")

            #  start session and train
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.local_variables_initializer(),
                          feed_dict={self.gpu_placeholders[self.num_gpu - 1]["is_train_placeholder"]: False})
            self.sess.run(tf.global_variables_initializer(),
                          feed_dict={self.gpu_placeholders[self.num_gpu - 1]["is_train_placeholder"]: False})

            if data_utils.get_latest_model(self.model_restore_path, MODEL_NAME_PREFIX):
                self._restore_from_checkpoint(self.sess)

            self.ops = [xy_loss, wh_loss, positive_conf_loss, negative_conf_loss,
                        cls_loss, loss, alpha_loss, hwl_loss, train_op]

    def reset_graph(self):
        """Reset default graph after training"""
        tf.reset_default_graph()

    def step(self, data):
        """
        Perform 1 update on the model with input training data.
        """
        feed_dict = {self.gpu_placeholders[self.num_gpu - 1]["is_train_placeholder"]: True}
        image_batch, label_batch_scale1, label_batch_scale2, label_batch_scale3, \
            cls_box_map_lists, objs_list, calib_list, _, _ = data
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        feed_dict.update({
            self.gpu_placeholders[self.num_gpu - 1]["input_image"]: (image_batch / 255.),
            self.gpu_placeholders[self.num_gpu - 1]["label_scale1"]: label_batch_scale1,
            self.gpu_placeholders[self.num_gpu - 1]["label_scale2"]: label_batch_scale2,
            self.gpu_placeholders[self.num_gpu - 1]["label_scale3"]: label_batch_scale3})

        xy_, wh_, positive_conf_, negative_conf_, cls_, loss_train, alpha_, hwl_, _ = \
            self.sess.run(self.ops, feed_dict=feed_dict, options=run_options)

        if self.cur_step % SUMMARY_INTERVAL == 0:
            feed_dict[self.gpu_placeholders[self.num_gpu - 1]["is_train_placeholder"]] = False
            xy_wh_conf_value = self.sess.run(self.xy_wh_conf, feed_dict=feed_dict)
            image_np = self._image_summary(image_batch, xy_wh_conf_value,
                                           calib_list, cls_box_map_lists)[0]
            feed_dict[self.gpu_placeholders[self.num_gpu - 1]["visual_image"]] = \
                np.expand_dims(image_np, axis=0)
            summary_train, summary_image = \
                self.sess.run([self.summary_op_train, self.image_summary_op],
                              feed_dict=feed_dict)
            self.summary_writer_train.add_summary(summary_train, global_step=self.cur_step)
            self.summary_writer_train.add_summary(summary_image, global_step=self.cur_step)
        if self.cur_step % PRINT_INTERVAL == 0:
            logging.info(f'step = {self.cur_step}, Loss = {loss_train}')
            logging.info(f'xy_loss = {xy_}, wh_loss = {wh_}, \
                          positive_conf_loss = {positive_conf_}, \
                          negative_conf_loss = {negative_conf_}, \
                          cls_loss = {cls_}, alpha_loss = {alpha_}, hwl_loss = {hwl_}.')

        print_current_memory_usage(f'current step: {self.cur_step}')

        # store the model every SAVE_INTERVAL epochs
        if self.cur_step % SAVE_INTERVAL == 0 or self.cur_step == MAX_ITER:
            self.saver.save(self.sess, "{}/{}".format(self.model_restore_path, MODEL_NAME_PREFIX),
                            global_step=self.cur_step)
            logging.info(f'Model saved in file: {self.model_restore_path}')
        logging.info(self.sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
        self.cur_step += 1

