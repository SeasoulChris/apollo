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
    def __init__(self, inducing_points, encoder_net_model, ard_num_dims, num_tasks):
        logging.info('inducing point size: {}'.format(inducing_points.shape))
        # mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2], batch_shape=torch.Size([num_tasks]))
        # wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal
        variational_strategy = MultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=num_tasks)

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        # kernel; added input dimension
        self.covar_module = ScaleKernel(
            MaternKernel(ard_num_dims=ard_num_dims, batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )  # default nu=2.5
        self.warping = encoder_net_model

    def forward(self, input_data):
        # warpped_input_date = self.warping(input_data)
        warpped_input_date = torch.cat(
            (self.warping(input_data[0, :]).unsqueeze(0),
             self.warping(input_data[1, :]).unsqueeze(0)), 0)
        # warpped_input_date = torch.cat(
        #     (self.warping(input_data[0, :]).unsqueeze(0),
        #      self.warping(input_data[1, :]).unsqueeze(0),
        #      self.warping(input_data[2, :]).unsqueeze(0)), 0)
        # logging.info(input_data.shape)
        mean_x = self.mean_module(warpped_input_date)
        covar_x = self.covar_module(warpped_input_date)
        return MultivariateNormal(mean_x, covar_x)


# class GPModelWithDerivatives(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMeanGrad()
#         self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
#         self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
