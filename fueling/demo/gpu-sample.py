"""A simple demo PySpark job."""
#!/usr/bin/env python

# Standard packages
import os
import subprocess
import time

# Third-party packages
from absl import flags
import colored_glog as glog
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

def print_nvidia_smi():
    """Print the output of nvidia-smi command"""
    glog.info('cuda available? {}'.format(torch.cuda.is_available()))
    glog.info('cuda version: {}'.format(torch.version.cuda))
    glog.info('gpu device count: {}'.format(torch.cuda.device_count()))
    check_output('nvidia-smi')

def run_torch_gpu_function(executor_name):
    """Run Pytorch training task with GPU option"""
    glog.info('current executor: {}'.format(executor_name))
    print_nvidia_smi()
    time_start = time.time()

    dtype = torch.float
    # device = torch.device("cpu")
    device = torch.device("cuda:0") # Uncomment this to run on GPU
        
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
    time.sleep(60 * 5)

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
        print_nvidia_smi()
        time_start = time.time()
        (self.to_rdd(['executor-1', 'executor-2', 'executor-3'])
         .foreach(run_torch_gpu_function))
        glog.info('Done with running Production, time spent: {}'.format(time.time() - time_start))

if __name__ == '__main__':
    GPUSample().main()
