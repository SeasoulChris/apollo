#!/usr/bin/env python


from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import gpytorch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import tqdm

from fueling.control.dynamic_model_2_0.gp_regression.encoder import DummyEncoder
import fueling.control.dynamic_model_2_0.gp_regression.train_utils as train_utils


# 3D data visualization
# from(https://github.com/krasserm/bayesian-machine-learning/blob
# /af6882305d9d65dbbf60fd29b117697ef250d4aa/gaussian_processes_util.py#L7)
def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm,
                    linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train)
    ax.set_title(title)


def func(x, y, z):
    """ joint distribution of y1 and y2 """
    return np.exp(x**2 - y**2 + z**2)**(0.05)


def func_y1(x, y):
    """ output function for y1"""
    return np.exp(x**2 - y**2)**(0.05)


def func_y2(x, z):
    """ output function for y2"""
    return np.exp(x**2 + z**2)**(0.05)


# ********** data preperation ********************
noise_2D = 0.1

rx, ry = np.arange(-5, 5, 0.3), np.arange(-5, 5, 0.3)
gx, gy = np.meshgrid(rx, rx)

X_2D = np.c_[gx.ravel(), gy.ravel()]

np_train_x = np.random.uniform(-4, 4, (500, 2))
train_x = torch.from_numpy(np_train_x).type(torch.FloatTensor)


np_train_y1 = func_y1(np_train_x[:, 0], np_train_x[:, 1])
np_train_y2 = func_y2(np_train_x[:, 0], np_train_x[:, 1])
# new data set
Y_2D_train_1 = torch.from_numpy(np_train_y1
                                + noise_2D
                                * np.random.randn(len(np_train_x))).type(torch.FloatTensor)
Y_2D_train_2 = torch.from_numpy(np_train_y2
                                + noise_2D
                                * np.random.randn(len(np_train_x))).type(torch.FloatTensor)
train_y = torch.stack([
    Y_2D_train_1,
    Y_2D_train_2,
], -1)


# exact
plt.figure(figsize=(14, 7))
exact_rx, exact_ry = np.arange(-4, 4, 0.1), np.arange(-4, 4, 0.1)
g_train_x1, g_train_x2 = np.meshgrid(exact_rx, exact_ry)
X_2D_train = np.c_[g_train_x1.ravel(), g_train_x2.ravel()]
exact_y1 = func_y1(X_2D_train[:, 0], X_2D_train[:, 1])
ax = plt.gcf().add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(g_train_x1, g_train_x2, exact_y1.reshape(g_train_x1.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(np_train_x[:, 0], np_train_x[:, 1], train_y[:, 0].detach().numpy())


exact_y2 = func_y2(X_2D_train[:, 0], X_2D_train[:, 1])
ax = plt.gcf().add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(g_train_x1, g_train_x2, exact_y2.reshape(g_train_x1.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(np_train_x[:, 0], np_train_x[:, 1], train_y[:, 1].detach().numpy())
plt.show()

print(f'X_2D_train shape is {train_x.shape}')
print(f'Y_2D_train shape is {train_y.shape}')

#  ********* training **************
inducing_points = torch.rand(2, 32, 2)


# confs
num_epochs = 14
lr = 0.1

encoder_net_model = DummyEncoder()
model, likelihood, optimizer, loss = train_utils.init_train(
    inducing_points, encoder_net_model, train_y.size(1), train_y.size(0), lr)

# Training loader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_x, train_y), batch_size=32)

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


# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.from_numpy(X_2D).type(torch.FloatTensor)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# ********** predict ***************
plt.figure(figsize=(14, 7))

ax = plt.gcf().add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(gx, gy, mean[:, 0].numpy().reshape(gx.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
# TODO(Shu): better confident region visualization in 3D
ax.plot_surface(gx, gy, upper[:, 0].numpy().reshape(gx.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
ax.plot_surface(gx, gy, lower[:, 0].numpy().reshape(gx.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(train_x[:, 0].detach().numpy(), train_x[:, 1].detach().numpy(),
           train_y[:, 0].detach().numpy())

ax = plt.gcf().add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(gx, gy, mean[:, 1].numpy().reshape(gx.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
ax.plot_surface(gx, gy, upper[:, 1].numpy().reshape(gx.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
ax.plot_surface(gx, gy, lower[:, 1].numpy().reshape(gx.shape), cmap=cm.coolwarm,
                linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(train_x[:, 0].detach().numpy(), train_x[:, 1].detach().numpy(),
           train_y[:, 1].detach().numpy())
plt.show()


#  accuracy
test_exact_y1 = func_y1(X_2D[:, 0], X_2D[:, 1])
test_exact_y2 = func_y2(X_2D[:, 0], X_2D[:, 1])
test_y_exact = np.stack([test_exact_y1, test_exact_y2], axis=1)

print(test_y_exact.shape)
print(mean.shape)
mse_x = mean_squared_error(mean[:, 0], test_y_exact[:, 0])
mse_y = mean_squared_error(mean[:, 1], test_y_exact[:, 1])
print(f'mse loss is {mse_x}')
print(f'mse loss is {mse_y}')
# mse loss is 0.035218787166998104
# mse loss is 0.9804320714474639
