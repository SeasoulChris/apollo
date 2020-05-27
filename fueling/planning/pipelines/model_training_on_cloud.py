#!/usr/bin/env python

# Standard packages
import os
import sys
import time

# Third-party packages
import torch

# Apollo-fuel packages
from fueling.common.base_pipeline import BasePipeline
import fueling.common.logging as logging


class PytorchTraining(BasePipeline):
    """Demo pipeline."""

    def run(self):
        """Run."""
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

        # TODO(yifei) add train code here


if __name__ == '__main__':
    PytorchTraining().main()
