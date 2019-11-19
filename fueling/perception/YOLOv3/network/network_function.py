#!/usr/bin/env python

import os

import numpy as np
import tensorflow as tf


np.random.seed(101)
slim = tf.contrib.slim


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
             [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]],
                               mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


class YOLOv3(object):
    """Structure of reseau neural YOLO3"""

    def __init__(self, x, anchors, num_classes, is_training=True):
        """
        Create the graph of the YOLOv3 model
        :param x: Placeholder for the input tensor: (normalised image (416, 416, 3)/255.)
        :param num_classes: Number of classes in the dataset
               if it isn't in the same folder as this code
        """
        self.input_placeholder = x
        self.num_classes = num_classes
        self.is_training = is_training
        self.anchors = anchors

    def _conv2d_fixed_padding(self, inputs, num_filters, kernel_size, strides=1, scope=None):
        """
        If strides>1, Perform padding to input first and then convolve,
        so that the output has resolution of input_resolution/stride.
        """
        if strides > 1:
            inputs = _fixed_padding(inputs, kernel_size)
        inputs = slim.conv2d(inputs, num_filters, kernel_size, stride=strides,
                             padding=('SAME' if strides == 1 else 'VALID'))
        return inputs

    def _darknet53_block(self, inputs, filters):
        """
        Resnet block used in Darknet.
        """
        shortcut = inputs
        inputs = self._conv2d_fixed_padding(inputs, filters, 1)
        inputs = self._conv2d_fixed_padding(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    def darknet53(self, inputs):
        """
        Builds Darknet-53 model.
        """
        inputs = self._conv2d_fixed_padding(inputs, 32, 3)
        inputs = self._conv2d_fixed_padding(inputs, 64, 3, strides=2)
        inputs = self._darknet53_block(inputs, 32)
        inputs = self._conv2d_fixed_padding(inputs, 128, 3, strides=2)

        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)

        inputs = self._conv2d_fixed_padding(inputs, 256, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)

        route_1 = inputs
        inputs = self._conv2d_fixed_padding(inputs, 512, 3, strides=2)

        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        route_2 = inputs
        inputs = self._conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)

        return route_1, route_2, inputs

    def _yolo_block(self, inputs, num_filters):
        """
        The convolution blocks between the backbone darknet and detection layers.
        """
        inputs = self._conv2d_fixed_padding(inputs, num_filters, 1)
        inputs = self._conv2d_fixed_padding(inputs, num_filters * 2, 3)
        inputs = self._conv2d_fixed_padding(inputs, num_filters, 1)
        inputs = self._conv2d_fixed_padding(inputs, num_filters * 2, 3)
        inputs = self._conv2d_fixed_padding(inputs, num_filters, 1)
        route = inputs
        inputs = self._conv2d_fixed_padding(inputs, num_filters * 2, 3)
        return route, inputs

    def _detection_layer(self, inputs, data_format, num_layers, num_anchors_per_cell=3):
        """
        predictions: (bs, cel_row, cel_col, num_anchors_per_cel*n)
        """
        predictions = slim.conv2d(inputs, num_anchors_per_cell * (num_layers),
                                  kernel_size=1, stride=1, normalizer_fn=None,
                                  activation_fn=None,
                                  biases_initializer=tf.zeros_initializer())
        return predictions

    def yolo_v3(self, num_layers, num_anchors_per_cell,
                weight_decay=0.0005, data_format='NHWC', reuse=False):
        """
        Creates YOLO v3 model.

        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
            Dimension batch_size may be undefined. The channel order is RGB.
        :param is_training: whether is training or not.
        :param data_format: data format NCHW or NHWC.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # it will be needed later on
        img_size = self.input_placeholder.get_shape().as_list()[1:3]

        # transpose the inputs to NCHW
        if data_format == 'NCHW':
            input_tensor = tf.transpose(self.input_placeholder, [0, 3, 1, 2])
        else:
            input_tensor = self.input_placeholder

        # normalize values to range [0..1]
        # input_tensor = input_tensor / 255

        # set batch norm params
        batch_norm_params = {
            'decay': 0.995,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': self.is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding],
                            data_format=data_format, reuse=None, scope=None):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                with tf.variable_scope("detector", reuse=reuse):
                    with tf.variable_scope('darknet-53', reuse=reuse):
                        route_1, route_2, inputs = self.darknet53(input_tensor)

                    with tf.variable_scope('yolo-v3'):
                        # scale1
                        route, inputs = self._yolo_block(inputs, 512)
                        feature1 = inputs
                        detect_1 = self._detection_layer(inputs, data_format,
                                                         num_layers, num_anchors_per_cell)
                        # (bs, cel_row, cel_col, num_anchors_per_cel*n)
                        detect_1 = tf.identity(detect_1, name='detect_1')

                        # scale2
                        inputs = self._conv2d_fixed_padding(route, 256, 1)
                        upsample_size = route_2.get_shape().as_list()
                        inputs = self._upsample(inputs, upsample_size, data_format)
                        inputs = tf.concat([inputs, route_2],
                                           axis=1 if data_format == 'NCHW' else 3)

                        route, inputs = self._yolo_block(inputs, 256)
                        feature2 = inputs
                        detect_2 = self._detection_layer(inputs, data_format,
                                                         num_layers, num_anchors_per_cell)
                        # (bs, cel_row, cel_col, num_anchors_per_cel*(10 + num_classes))
                        detect_2 = tf.identity(detect_2, name='detect_2')

                        # scale3
                        inputs = self._conv2d_fixed_padding(route, 128, 1)
                        upsample_size = route_1.get_shape().as_list()
                        inputs = self._upsample(inputs, upsample_size, data_format)
                        inputs = tf.concat([inputs, route_1],
                                           axis=1 if data_format == 'NCHW' else 3)

                        _, inputs = self._yolo_block(inputs, 128)
                        feature3 = inputs
                        detect_3 = self._detection_layer(inputs, data_format,
                                                         num_layers, num_anchors_per_cell)
                        # (bs, cel_row, cel_col, num_anchors_per_cel*(10 + num_classes))
                        detect_3 = tf.identity(detect_3, name='detect_3')
        return detect_1, detect_2, detect_3, feature1, feature2, feature3

    def _upsample(self, inputs, out_shape, data_format='NCHW'):
        """
        Perform upsample on the inputs.
        """
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        if data_format == 'NCHW':
            new_height = out_shape[3]
            new_width = out_shape[2]
        elif data_format == 'NHWC':
            new_height = out_shape[1]
            new_width = out_shape[2]

        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

        # back to NCHW if needed
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = tf.identity(inputs, name='upsampled')
        return inputs
