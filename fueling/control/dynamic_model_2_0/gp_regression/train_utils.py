#!/usr/bin/env python


from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import gpytorch
import torch
import tqdm

from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
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


def basic_train_loop(train_loader, model, loss, optimizer, is_transpose=False):
    loss_history = []
    for x_batch, y_batch in train_loader:
        # **[window_size, batch_size, channel]
        # training process
        optimizer.zero_grad()
        if is_transpose:
            x_batch = torch.transpose(x_batch, 0, 1).type(torch.FloatTensor)
        output = model(x_batch)
        # train loss
        train_loss = -loss(output, y_batch)
        train_loss.backward()
        optimizer.step()
        loss_history.append(train_loss.item())
    loss_history_mean = np.mean(loss_history)
    print(f'train loss is {loss_history_mean}')
    return loss_history_mean


def train_with_adjusted_lr(num_epochs, train_loader, model, likelihood,
                           loss, optimizer, is_transpose=False):
    # adjust learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                  verbose=False, threshold=0.0001, threshold_mode='rel',
                                  cooldown=0, min_lr=0.0, eps=1e-08)

    # training
    model.train()
    likelihood.train()

    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        train_loss = basic_train_loop(train_loader, model, loss, optimizer, is_transpose)
        scheduler.step(train_loss)
        if i == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)
    return model, likelihood


def load_model(file_path, encoder_net_model, model, likelihood):
    """
    load state dict for model and likelihood
    """
    # load state dict
    logging.info(f"Loading GP model from {file_path}")
    model_state_dict, likelihood_state_dict = torch.load(file_path)
    model.load_state_dict(model_state_dict)
    likelihood.load_state_dict(likelihood_state_dict)
