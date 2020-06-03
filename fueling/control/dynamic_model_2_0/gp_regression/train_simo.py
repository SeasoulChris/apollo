#!/usr/bin/env python

import math


from matplotlib import pyplot as plt
import gpytorch
import numpy as np
import torch
import torch.nn as nn
import tqdm

from fueling.control.dynamic_model_2_0.gp_regression.encoder import DummyEncoder
import fueling.control.dynamic_model_2_0.gp_regression.train_utils as train_utils


# based on https://gpytorch.readthedocs.io/en/latest/examples/04_Variational_and_Approximate_GPs
# /SVGP_Multitask_GP_Regression.html#Set-up-training-data
# confs
num_epochs = 20
lr = 0.01

#   load data
train_x = torch.linspace(0, 1, 100)

train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
], -1)

inducing_points = torch.rand(2, 16, 1)

# encoder
encoder_net_model = DummyEncoder()
model, likelihood, optimizer, loss = train_utils.init_train(
    inducing_points, encoder_net_model, train_y.size(1), train_y.size(0), lr)

# Training loader (different data loader)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_x, train_y))

model.train()
likelihood.train()

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    train_utils.basic_train_loop(train_loader, model, loss, optimizer)


# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task

# Plot training data as black stars
y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
# Predictive mean as blue line
y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
# Shade in confidence
y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
y1_ax.set_ylim([-3, 3])
y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
y2_ax.set_ylim([-3, 3])
y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y2_ax.set_title('Observed Values (Likelihood)')
plt.show()
