#!/usr/bin/env python
"""
A simple demo PySpark job with GPU training.

Run with:
    ./tools/submit-job-to-k8s.sh --gpu --env fuel-py36 fueling/demo/gpu_sample.py
"""

# Standard packages
import subprocess
import time

# Third-party packages
from tensorflow.python.client import device_lib
import colored_glog as glog
import tensorflow as tf
import torch

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline


def check_output(command):
    """Return the output of given system command"""
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_lines = proc.stdout.readlines()
    proc.communicate()
    for line in out_lines:
        glog.info(line.strip())


def run_tensorflow_gpu_function(executor_name):
    """Run tensorflow training task"""
    glog.info('current executor: {}'.format(executor_name))
    check_output('nvidia-smi')

    if tf.test.gpu_device_name():
        glog.info('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        glog.info('GPU available? {}'.format(tf.test.is_gpu_available()))
        glog.info('GPU devices: {}'.format(device_lib.list_local_devices()))
    else:
        glog.warn('Please install GPU version of TF')

    time_start = time.time()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data('/mnt/bos/test/datasets/mnist.npz')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

    glog.info('Tensorflow GPU function is done, time spent: {}'.format(time.time() - time_start))
    time.sleep(60 * 3)


def run_torch_gpu_function(executor_name):
    """Run Pytorch training task with GPU option"""
    glog.info('current executor: {}'.format(executor_name))
    check_output('nvidia-smi')

    glog.info('cuda available? {}'.format(torch.cuda.is_available()))
    glog.info('cuda version: {}'.format(torch.version.cuda))
    glog.info('gpu device count: {}'.format(torch.cuda.device_count()))

    time_start = time.time()

    dtype = torch.float
    # Or use 'cpu' for CPU training.
    device = torch.device('cuda:0')

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        glog.info(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    glog.info('Torch GPU function is done, time spent: {}'.format(time.time() - time_start))


class GPUSample(BasePipeline):
    """Demo pipeline."""

    def __init__(self):
        BasePipeline.__init__(self, 'gpu sample demo')

    def run_test(self):
        """Run test."""
        glog.info('not designed for test')
        return

    def run_prod(self):
        """Run prod."""
        glog.info('Running Production')
        check_output('nvidia-smi')
        time_start = time.time()

        self.to_rdd(['executor-pytorch']).foreach(run_torch_gpu_function)
        self.to_rdd(['executor-tensorflow']).foreach(run_tensorflow_gpu_function)
        glog.info('Done with running Production, time spent: {}'.format(time.time() - time_start))


if __name__ == '__main__':
    GPUSample().main()
