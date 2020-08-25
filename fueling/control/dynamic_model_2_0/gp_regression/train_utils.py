#!/usr/bin/env python
import os
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import gpytorch
import torch
import tqdm

import matplotlib.pyplot as plt

from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_state_dict
from fueling.control.dynamic_model_2_0.gp_regression.train import save_model_torch_script
from fueling.learning.train_utils import cuda
import fueling.common.logging as logging


def init_train(inducing_points, encoder_net_model, output_dim, total_train_number, lr, kernel_dim):
    # model
    model = GPModel(inducing_points=inducing_points,
                    encoder_net_model=encoder_net_model, ard_num_dims=kernel_dim,
                    num_tasks=output_dim)
    #  likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)

    # optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    loss = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=total_train_number)

    return model, likelihood, optimizer, loss


def basic_train_loop(train_loader, model, loss_fn, optimizer, is_transpose=False, use_cuda=False):
    loss_history = []
    for x_batch, y_batch in train_loader:
        # **[window_size, batch_size, channel]
        # training process
        optimizer.zero_grad()
        if is_transpose:
            x_batch = torch.transpose(x_batch, 0, 1).type(torch.FloatTensor)
        if use_cuda:
            x_batch, y_batch = cuda(x_batch), cuda(y_batch)
        output = model(x_batch)
        # train loss
        train_loss = -loss_fn(output, y_batch)
        train_loss.backward()
        optimizer.step()
        loss_history.append(train_loss.item())
    loss_history_mean = np.mean(loss_history)
    print(f'train loss is {loss_history_mean}')
    return loss_history_mean


def train_with_adjusted_lr(num_epochs, train_loader, model, likelihood,
                           loss_fn, optimizer, fig_file_path=None, is_transpose=False,
                           use_cuda=False):
    # adjust learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                  verbose=False, threshold=0.0001, threshold_mode='rel',
                                  cooldown=0, min_lr=0.0, eps=1e-08)

    # training
    model.train()
    likelihood.train()

    # save train loss
    train_loss_all = []
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        train_loss = basic_train_loop(train_loader, model, loss_fn,
                                      optimizer, is_transpose, use_cuda)
        scheduler.step(train_loss)
        train_loss_all.append(train_loss)
        if i == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)
    plot_train_loss(train_loss_all, fig_file_path)
    # output last train loss
    return model, likelihood, train_loss_all[-1]


def train_with_cross_validation(num_epochs, train_loader, validation_loader, model, likelihood,
                                loss_fn, optimizer, fig_file_path=None, is_transpose=False,
                                use_cuda=False):
    # adjust learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                  verbose=False, threshold=0.0001, threshold_mode='rel',
                                  cooldown=0, min_lr=0.0, eps=1e-08)

    # save train loss
    loss_dict = dict()
    loss_dict["train_loss_all"] = []
    loss_dict["validation_mean_loss_all"] = []
    loss_dict["validation_mean_accuracy_all"] = []
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        # training loop
        model.train()
        likelihood.train()
        train_loss = basic_train_loop(train_loader, model, loss_fn,
                                      optimizer, is_transpose, use_cuda)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # validation loop
            validation_mean_loss, validation_mean_accuracy = validation_loop(
                validation_loader, model, likelihood, loss_fn,
                accuracy_fn=torch.nn.MSELoss(), is_transpose=is_transpose, use_cuda=use_cuda)
        scheduler.step(validation_mean_loss)
        if i == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)
        loss_dict["train_loss_all"].append(train_loss)
        loss_dict["validation_mean_loss_all"].append(validation_mean_loss)
        loss_dict["validation_mean_accuracy_all"].append(validation_mean_accuracy)
    plot_loss(loss_dict, fig_file_path)
    return model, likelihood


def train_save_best_model(num_epochs, train_loader, model, likelihood,
                          loss_fn, optimizer, test_features, result_folder,
                          fig_file_path=None, is_transpose=False):
    # adjust learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                  verbose=False, threshold=0.0001, threshold_mode='rel',
                                  cooldown=0, min_lr=0.0, eps=1e-08)

    # training
    model.train()
    likelihood.train()

    # save train loss
    best_train_loss = float('+inf')
    train_loss_all = []
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    time_start = time.time()
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        train_loss = basic_train_loop(train_loader, model, loss_fn, optimizer, is_transpose)
        scheduler.step(train_loss)
        train_loss_all.append(train_loss)
        if i == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)

        is_better_model = False
        if train_loss < best_train_loss:
            num_epoch_valid_loss_not_decreasing = 0
            best_train_loss = train_loss
            is_better_model = True
        else:
            num_epoch_valid_loss_not_decreasing += 1

        if is_better_model:
            # save mode
            model.eval()
            likelihood.eval()
            offline_model_path = os.path.join(result_folder, f'{i}', 'gp_model.pth')
            online_model_path = os.path.join(result_folder, f'{i}', 'gp_model.pt')
            save_model_state_dict(model, likelihood, offline_model_path)
            plot_train_loss(train_loss_all, fig_file_path)
            save_model_torch_script(model, likelihood, test_features, online_model_path)
            likelihood.train()
            model.train()

        logging.info(
            f'epoch {i} cost time is : {time.time() - time_start} and current loss is {train_loss}')
        time_start = time.time()

    plot_train_loss(train_loss_all, fig_file_path)
    # output last train loss
    return model, likelihood, train_loss_all[-1]


def validation_loop(data_loader, model, likelihood, loss_fn, accuracy_fn=torch.nn.MSELoss(),
                    is_transpose=False, use_cuda=False):
    """ training with validation loss """
    model.eval()
    likelihood.eval()
    loss_history = []
    accuracy_history = []
    # loop over batches
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if is_transpose:
            x_batch = torch.transpose(x_batch, 0, 1)
        if use_cuda:
            x_batch, y_batch = cuda(x_batch), cuda(y_batch)
        # same as training loss formation
        output = model(x_batch)
        loss = -loss_fn(output, y_batch)
        loss_history.append(loss.item())
        # evaluation accuracy
        predictions = likelihood(model(x_batch))
        mean = predictions.mean
        # lower, upper = predictions.confidence_region()
        accuracy = accuracy_fn(mean, y_batch)
        accuracy_history.append(accuracy.item())

    mean_loss = np.mean(loss_history)
    mean_accuracy = np.mean(accuracy_history)
    logging.info(f'loss: {mean_loss}.')
    logging.info(f'accuracy: {mean_accuracy}')
    return mean_loss, mean_accuracy


def plot_train_loss(train_losses, fig_file_path):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('epoch', fontdict={'size': 12})
    ax1.set_ylabel('loss', fontdict={'size': 12})
    ax1.plot(train_losses, label="training loss")
    plt.legend(fontsize=12, numpoints=5, frameon=False)
    plt.title("Training Loss")
    plt.grid(True)
    if fig_file_path is not None:
        plt.savefig(fig_file_path)


def plot_loss(loss_dict, fig_file_path):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('epoch', fontdict={'size': 12})
    ax1.set_ylabel('loss', fontdict={'size': 12})
    for key in loss_dict:
        ax1.plot(loss_dict[key], label=key)
    plt.legend(fontsize=12, numpoints=5, frameon=False)
    plt.title("Loss and Accuracy")
    plt.grid(True)
    if fig_file_path is not None:
        plt.savefig(fig_file_path)


def load_model(file_path, encoder_net_model, model, likelihood):
    """
    load state dict for model and likelihood
    """
    # load state dict
    logging.info(f"Loading GP model from {file_path}")
    model_state_dict, likelihood_state_dict = torch.load(file_path)
    model.load_state_dict(model_state_dict)
    likelihood.load_state_dict(likelihood_state_dict)
