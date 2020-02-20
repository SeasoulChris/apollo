#!/usr/bin/env python
"""
A simple demo PySpark job with GPU training.

Run with:
    ./tools/submit-job-to-k8s.py --main=fueling/demo/gpu_training_with_pytorch.py \
        --node_selector=GPU
"""

# Standard packages
import os
import sys
import time

# Third-party packages
import torch

# Apollo-fuel packages
from fueling.common.base_pipeline_v2 import BasePipelineV2
import fueling.common.logging as logging


class PytorchTraining(BasePipelineV2):
    """Demo pipeline."""

    def run(self):
        """Run test."""
        logging.info('nvidia-smi on Driver:')
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        time_start = time.time()
        self.to_rdd(range(1)).foreach(self.train)
        logging.info('Training complete in {} seconds.'.format(time.time() - time_start))

    @staticmethod
    def train(instance_id):
        """Run training task"""
        logging.info('nvidia-smi on Executor {}:'.format(instance_id))
        if os.system('nvidia-smi') != 0:
            logging.fatal('Failed to run nvidia-smi.')
            sys.exit(-1)

        logging.info('cuda available? {}'.format(torch.cuda.is_available()))
        logging.info('cuda version: {}'.format(torch.version.cuda))
        logging.info('gpu device count: {}'.format(torch.cuda.device_count()))

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
        for t in range(100):
            # Forward pass: compute predicted y
            h = x.mm(w1)
            h_relu = h.clamp(min=0)
            y_pred = h_relu.mm(w2)

            # Compute and print loss
            loss = (y_pred - y).pow(2).sum().item()
            logging.info(t, loss)

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


if __name__ == '__main__':
    PytorchTraining().main()
