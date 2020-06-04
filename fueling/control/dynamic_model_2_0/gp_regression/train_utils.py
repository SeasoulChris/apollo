#!/usr/bin/env python


from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import numpy as np
import gpytorch
import torch

from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
import fueling.common.logging as logging


def init_train(inducing_points, encoder_net_model, output_dim, total_train_number, lr):
    # model
    model = GPModel(inducing_points=inducing_points,
                    encoder_net_model=encoder_net_model, num_tasks=output_dim)
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
    train_loss = np.mean(loss_history)
    logging.info(f'train loss is {train_loss}')
    return train_loss
