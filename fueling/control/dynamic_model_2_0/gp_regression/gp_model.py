#!/usr/bin/env python

from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.kernels import MaternKernel, InducingPointKernel, ScaleKernel
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
        # fake_inducing_points = torch.randn(2, 16, 100, 6)
        variational_strategy = MultitaskVariationalStrategy(VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        ), num_tasks=num_tasks)

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        # kernel
        self.covar_module = ScaleKernel(MaternKernel(
            batch_shape=torch.Size([num_tasks])), batch_shape=torch.Size([num_tasks]))  # default nu=2.5
        self.warping = encoder_net_model

    def forward(self, input_data):
        logging.debug(f'input data is {input_data[-1, -1,:]}')
        # warpped_input_date = self.warping(input_data)
        warpped_input_date = self.warping(input_data[0, :])
        for i in range(1, input_data.shape[0]):
            torch.cat((warpped_input_date, self.warping(input_data[i, :])), 0)
        logging.debug(f'input data is {input_data.shape}')
        mean_x = self.mean_module(warpped_input_date)
        logging.debug(f'input data mean value is {mean_x[:, 0]}')
        covar_x = self.covar_module(warpped_input_date)
        return MultivariateNormal(mean_x, covar_x)


class GPModelLayer(ApproximateGP):
    def __init__(self, input_dims, output_dims, num_inducing):
        batch_shape = torch.Size([output_dims])
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape)
        variational_strategy = MultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=batch_shape)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.base_kernel = MaternKernel(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(self.base_kernel, batch_shape=batch_shape)

    def forward(self, input_data):
        mean_x = self.mean_module(input_data)
        covar_x = self.covar_module(input_data)
        return MultivariateNormal(mean_x, covar_x)
