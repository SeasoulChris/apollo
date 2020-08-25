#!/usr/bin/env python

import math


from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import gpytorch
import torch
import torch.utils.data

from fueling.control.dynamic_model_2_0.gp_regression.encoder import DummyEncoder
import fueling.control.dynamic_model_2_0.gp_regression.train_utils as train_utils


# based on https://gpytorch.readthedocs.io/en/latest/examples/04_Variational_and_Approximate_GPs
# /SVGP_Multitask_GP_Regression.html#Set-up-training-data
# confs
num_epochs = 50
lr = 0.01
num_inducing_point = 16
kernal_dim = 1
output_dim = 2
train_data_nums = 100
#   load data
train_x = torch.linspace(0, 1, 100)


train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
], -1)

inducing_points = torch.rand(2, num_inducing_point, 1)

# encoder
encoder_net_model = DummyEncoder()
# use scale constant other than torch dimension
model, likelihood, optimizer, loss = train_utils.init_train(
    inducing_points, encoder_net_model, output_dim, train_data_nums, lr,
    kernel_dim=kernal_dim)

# Training loader (different data loader)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_x, train_y))

model, likelihood, train_loss_all = train_utils.train_with_adjusted_lr(
    num_epochs, train_loader, model, likelihood,
    loss, optimizer)

# validation loader
test_x = torch.linspace(0, 1, 51)
test_y = torch.stack([
    torch.sin(test_x * (2 * math.pi)) + torch.randn(test_x.size()) * 0.2,
    torch.cos(test_x * (2 * math.pi)) + torch.randn(test_x.size()) * 0.2,
], -1)
validation_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_x, test_y))
with torch.no_grad():
    mean_loss, mean_accuracy = train_utils.validation_loop(
        validation_loader, model, likelihood, loss_fn=loss, accuracy_fn=torch.nn.MSELoss(),
        is_transpose=False, use_cuda=False)

print(f'validation mean loss is: {mean_loss}; mean accuracy is: {mean_accuracy}')
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


#  accuracy
test_y_exact = torch.stack([
    torch.sin(test_x * (2 * math.pi)),
    torch.cos(test_x * (2 * math.pi)),
], -1).detach().numpy()
mse_x = mean_squared_error(mean[:, 0], test_y_exact[:, 0])
mse_y = mean_squared_error(mean[:, 1], test_y_exact[:, 1])
print(f'mse loss is {mse_x}')
print(f'mse loss is {mse_y}')
# mse loss is 0.023380782455205917
# mse loss is 0.07363846898078918
