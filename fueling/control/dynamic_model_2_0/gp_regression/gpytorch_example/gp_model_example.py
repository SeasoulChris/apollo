#!/usr/bin/env python

from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.kernels import MaternKernel, InducingPointKernel
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch

import fueling.common.logging as logging


class GPModelExample(ApproximateGP):
    def __init__(self, inducing_points, input_data_dim):
        logging.info('inducing point size: {}'.format(inducing_points.shape))
        # (100*6) input dimension
        variational_distribution = CholeskyVariationalDistribution(inducing_points.shape[0])
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        # logging.info(inducing_points)
        super(GPModelExample, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # kernel
        self.covar_module = MaternKernel()  # default nu=2.5
        # SparseGPR
        # self.covar_module = InducingPointKernel(
        #     self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, input_data):
        mean_x = self.mean_module(input_data)
        covar_x = self.covar_module(input_data)
        return MultivariateNormal(mean_x, covar_x)
