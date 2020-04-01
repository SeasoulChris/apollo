#!/usr/bin/env python

from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.kernels import MaternKernel, InducingPointKernel
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import MultitaskVariationalStrategy, VariationalStrategy
import gpytorch
import torch

from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
import fueling.common.logging as logging


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, encoder_net_model, num_tasks):
        logging.info('inducing point size: {}'.format(inducing_points.shape))
        # mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2], batch_shape=torch.Size([num_tasks]))
        # wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal
        variational_strategy = MultitaskVariationalStrategy(VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        ), num_tasks=num_tasks)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        # kernel
        self.covar_module = MaternKernel(batch_shape=torch.Size([num_tasks]))  # default nu=2.5
        self.warping = encoder_net_model

    def forward(self, input_data):
        input_data = self.warping(input_data)
        mean_x = self.mean_module(input_data)
        covar_x = self.covar_module(input_data)
        return MultivariateNormal(mean_x, covar_x)
