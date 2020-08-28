#!/usr/bin/env python
"""Training utils."""

import math
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

import fueling.common.logging as logging


#########################################################
# Helper functions
#########################################################
def cuda(x):
    if isinstance(x, list):
        return [cuda(y) for y in x]
    if isinstance(x, tuple):
        return tuple([cuda(y) for y in x])
    return x.cuda() if torch.cuda.is_available() else x


# TODO(jiacheng): implement this


def save_checkpoint():
    return


#########################################################
# Vanilla training and validating
#########################################################


def train_vanilla(train_X, train_y, model, loss, optimizer, epoch,
                  batch_preprocess=None, batch_size=1024, print_period=100):
    model.train()

    loss_history = []
    logging.info('Epoch: {}:'.format(epoch))
    num_of_data = train_X.shape[0]
    num_of_batch = math.ceil(num_of_data / batch_size)
    pred_y = []
    for i in range(num_of_batch):
        optimizer.zero_grad()
        X = train_X[i * batch_size: min(num_of_data, (i + 1) * batch_size), ]
        y = train_y[i * batch_size: min(num_of_data, (i + 1) * batch_size), ]
        if batch_preprocess is not None:
            X, y = batch_preprocess(X, y)
        pred = model(X)
        train_loss = loss.loss_fn(pred, y)
        loss_history.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        pred = pred.detach().cpu().numpy()
        pred_y.append(pred)

        if (i > 0) and (i % print_period == 0):
            logging.info('   Step: {}, training loss: {}'.format(
                i, np.mean(loss_history[-print_period:])))

    train_loss = np.mean(loss_history)
    logging.info('Training loss: {}'.format(train_loss))
    loss.loss_info(pred_y, train_y)


def valid_vanilla(valid_X, valid_y, model, loss, batch_preprocess=None,
                  batch_size=1024):
    model.eval()

    loss_history = []
    num_of_data = valid_X.shape[0]
    num_of_batch = math.ceil(num_of_data / batch_size)
    pred_y = []
    for i in range(num_of_batch):
        X = valid_X[i * batch_size: min(num_of_data, (i + 1) * batch_size), ]
        y = valid_y[i * batch_size: min(num_of_data, (i + 1) * batch_size), ]
        if batch_preprocess is not None:
            X, y = batch_preprocess(X, y)
        pred = model(X)
        valid_loss = loss.loss_fn(pred, y)
        loss_history.append(valid_loss.item())

        pred = pred.detach().cpu().numpy()
        pred_y.append(pred)

    valid_loss = np.mean(loss_history)
    logging.info('Validation loss: {}.'.format(valid_loss))
    loss.loss_info(pred_y, valid_y)

    return valid_loss


def train_valid_vanilla(train_X, train_y, valid_X, valid_y, model, loss, optimizer,
                        scheduler, epochs, save_name, batch_preprocess=None,
                        train_batch=1024, print_period=100, valid_batch=1024):
    best_valid_loss = float('+inf')
    for epoch in range(1, epochs + 1):
        train_vanilla(train_X, train_y, model, loss, optimizer, epoch,
                      batch_preprocess, train_batch, print_period)
        valid_loss = valid_vanilla(valid_X, valid_y, model, loss,
                                   batch_preprocess, valid_batch)

        scheduler.step(valid_loss)

        # TODO(jiacheng): add early stopping mechanism
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # save_checkpoint()

    return model

#########################################################
# Training and validating using data-loader
#########################################################


def train_dataloader(train_loader, model, loss, optimizer, epoch,
                     print_period=None, visualizer=None):
    model.train()

    loss_history = []
    logging.info('Epoch: {}:'.format(epoch))
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X, y = cuda(X), cuda(y)
        pred = model(X)
        train_loss = loss.loss_fn(pred, y)

        train_total_loss = 0.0
        if isinstance(train_loss, dict):
            train_total_loss = train_loss["total_loss"]
            for key, value in train_loss.items():
                if key == "total_loss" and visualizer is not None:
                    visualizer.add_scalar("Train Total Loss/total loss(every batch)",
                                          value, (epoch - 1) * len(train_loader) + i)
                elif visualizer is not None:
                    visualizer.add_scalar("Train Loss/" + key + "(every batch)",
                                          value, (epoch - 1) * len(train_loader) + i)
        else:
            train_total_loss = train_loss
            if visualizer is not None:
                visualizer.add_scalar("Train Total Loss/total loss(every batch)",
                                      train_total_loss, (epoch - 1) * len(train_loader) + i)

        loss_history.append(train_total_loss.item())
        train_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        if print_period is None:
            continue
        if i > 0 and i % print_period == 0:
            logging.info('   Step: {}, training loss: {}'.format(
                i, np.mean(loss_history[-print_period:])))

    train_loss = np.mean(loss_history)
    logging.info('Training loss: {}'.format(train_loss))


def valid_dataloader(valid_loader, model, loss, epoch, visualizer=None, analyzer=None):
    model.eval()

    loss_history = []
    loss_info_history = []
    for i, (X, y) in enumerate(valid_loader):
        X, y = cuda(X), cuda(y)
        pred = model(X)
        valid_loss = loss.loss_fn(pred, y)

        valid_total_loss = 0.0
        if isinstance(valid_loss, dict):
            valid_total_loss = valid_loss["total_loss"]
        else:
            valid_total_loss = valid_loss

        loss_history.append(valid_total_loss.item())

        valid_loss_info = loss.loss_info(pred, y)
        if valid_loss_info is not None:
            loss_info_history.append(valid_loss_info.item())
        if analyzer is not None:
            analyzer.process(X, y, pred)

    valid_mean_loss = np.mean(loss_history)
    visualizer.add_scalar("Valid Loss/val mean loss(every epoch)", valid_mean_loss, epoch - 1)

    logging.info('Validation loss: {}.'.format(valid_mean_loss))
    logging.info('Validation accuracy = {}'.format(np.mean(loss_info_history)))

    return np.mean(loss_info_history)


def train_valid_dataloader(train_loader, valid_loader, model, loss, optimizer,
                           scheduler, epochs, save_name, print_period=None,
                           early_stop=None, save_mode=1, model_type="vis_log", visualize_dir=None):
    '''
        -save_mode: 0 - save the best one only
                    1 - save all models that are better than before
                    2 - save all
    '''

    # Tensorboard, visualize losses.
    '''
        tensorboard usage:
            in sever container, input command: tensorboard --logdir=visualize_dir
            in local host, open browser: server_ip:6006
    '''
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d_%H:%M:%S')
    model_name = model_type + '-' + time_str
    logdir = os.path.join(visualize_dir, model_name)
    Writer = SummaryWriter(log_dir=logdir)

    best_valid_loss = float('+inf')
    num_epoch_valid_loss_not_decreasing = 0
    for epoch in range(1, epochs + 1):
        train_dataloader(train_loader, model, loss, optimizer, epoch,
                         print_period, visualizer=Writer)
        with torch.no_grad():
            valid_loss = valid_dataloader(valid_loader, model, loss, epoch, visualizer=Writer)
        scheduler.step(valid_loss)

        # Determine if valid_loss is getting better and if early_stop is needed.
        is_better_model = False
        if valid_loss < best_valid_loss:
            num_epoch_valid_loss_not_decreasing = 0
            best_valid_loss = valid_loss
            is_better_model = True
        else:
            num_epoch_valid_loss_not_decreasing += 1
            # Early stop if enabled and met the criterion
            if early_stop == num_epoch_valid_loss_not_decreasing:
                logging.info('Reached early-stopping criterion. Stop training.')
                logging.info('Best validation loss = {}'.format(best_valid_loss))
                break

        # Save model according to the specified mode.
        epoch_model_name = 'model_epoch{}_valloss{:.6f}.pt'.format(epoch, valid_loss)
        if save_mode == 0:
            if is_better_model:
                torch.save(model.state_dict(), os.path.join(save_name, 'model.pt'))
        elif save_mode == 1:
            if is_better_model:
                torch.save(model.state_dict(), os.path.join(save_name, epoch_model_name))
        elif save_mode == 2:
            torch.save(model.state_dict(), os.path.join(save_name, epoch_model_name))

    return model
