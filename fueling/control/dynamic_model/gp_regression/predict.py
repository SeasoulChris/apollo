#!/usr/bin/env python

from collections import defaultdict
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
import warnings
from pyro.contrib.gp.parameterized import Parameterized
import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer
import pyro.optim as optim
import torch

# assert pyro.__version__.startswith('1.0.0')


class Predict(torch.nn.Module):
    def __init__(self, model, guide):
        super().__init__()
        self.model = model
        self.guide = guide

    def forward(self, *args, **kwargs):
        samples = {}
        guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(poutine.replay(
            self.model, guide_trace)).get_trace(*args, **kwargs)
        for site in prune_subsample_sites(model_trace).stochastic_nodes:
            samples[site] = model_trace.nodes[site]['value']
        return tuple(v for _, v in sorted(samples.items()))
