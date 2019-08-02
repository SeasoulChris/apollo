#!/usr/bin/env python

"""Utility functions"""

from pyro.contrib.gp.likelihoods import Likelihood
from torch.distributions import constraints
from torch.nn import Parameter
import pyro
import pyro.distributions as dist
import torch

def jacobian(inputs, outputs):
	"""Compute the Jacobian of the nonlinear system"""
	J = torch.zeros(outputs.shape[0], inputs.shape[0])
	for i in range(outputs.shape[0]):
		J_row =  torch.autograd.grad([outputs[i]], [inputs], retain_graph=True,
									 create_graph=True, allow_unused=True)[0]
		if not J_row is None:
			J[i] = J_row
	return J
