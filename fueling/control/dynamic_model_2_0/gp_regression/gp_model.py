#!/usr/bin/env python

from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.kernels import MaternKernel, InducingPointKernel
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch

from fueling.control.dynamic_model_2_0.gp_regression.encoder_gpytorch import GPEncoder
import fueling.common.logging as logging


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, input_data_dim):
        logging.info('inducing point size: {}'.format(inducing_points.shape))
        # (100*6) input dimension
        variational_distribution = CholeskyVariationalDistribution(inducing_points.shape[0])
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        # logging.info(inducing_points)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # kernel
        self.covar_module = MaternKernel()  # default nu=2.5
        self.warping = GPEncoder(input_data_dim, kernel_dim=20)
        # SparseGPR
        # self.covar_module = InducingPointKernel(
        #     self.base_covar_module, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, input_data):
        logging.debug('input data size before encoding: {}'.format(input_data.shape))
        input_data = self.warping(input_data)
        logging.debug('input data size after encoding: {}'.format(input_data.shape))
        logging.debug(input_data.shape[-1])
        mean_x = self.mean_module(input_data)
        covar_x = self.covar_module(input_data)
        return MultivariateNormal(mean_x, covar_x)
