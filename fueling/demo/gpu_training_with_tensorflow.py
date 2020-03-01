#!/usr/bin/env python
"""
A simple demo PySpark job with GPU training.

Run with:
    ./tools/submit-job-to-k8s.py --main=fueling/demo/gpu_training_with_tensorflow.py \
        --node_selector=GPU
"""

# Standard packages
import os
import sys
import time

# Third-party packages
from tensorflow.python.client import device_lib
import tensorflow as tf

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class TensorflowTraining(BasePipeline):
    """Demo pipeline."""

    def run_test(self):
        """Run test."""
        logging.info('nvidia-smi on Driver:')
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.train)
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))

    def run(self):
        """Same with run_test()."""
        self.run_test()

    @staticmethod
    def train(instance_id):
        """Run training task"""
        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        if tf.test.gpu_device_name():
            logging.info('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
            logging.info('GPU available? {}'.format(tf.test.is_gpu_available()))
            logging.info('GPU devices: {}'.format(device_lib.list_local_devices()))
        else:
            logging.fatal('Cannot access GPU.')
            sys.exit(-1)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data('/mnt/bos/test/datasets/mnist.npz')
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test)


if __name__ == '__main__':
    TensorflowTraining().main()
