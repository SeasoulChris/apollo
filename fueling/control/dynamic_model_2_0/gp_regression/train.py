#!/usr/bin/env python
import argparse
import os
import time

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import gpytorch
import torch
import torch.nn as nn

from fueling.control.dynamic_model_2_0.conf.model_conf import feature_config
from fueling.control.dynamic_model_2_0.gp_regression.dynamic_model_dataset import \
    DynamicModelDataset
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.gp_model import GPModel
import fueling.common.logging as logging


INPUT_DIM = feature_config["input_dim"]
OUTPUT_DIM = feature_config["output_dim"]


def accuracy(ground_truth, model_output, accuracy_history):
    criterion = nn.MSELoss()
    accuracy_info = torch.sqrt(criterion(ground_truth, model_output))
    if accuracy_info is not None:
        accuracy_history.append(accuracy_info.item())


def result_tracing(data_loader, model, likelihood):
    accuracies = []
    for idx, (features, labels) in enumerate(data_loader):
        features = torch.transpose(features, 0, 1).type(torch.FloatTensor)
        pred = likelihood(model(features))
        logging.info(f'features {features[0, 0, :]}')
        # logging.info(f'label {labels[0, :]}')
        # logging.info(f'predict {pred.mean[0, :]}')
        accuracy(labels, pred.mean, accuracies)
    return np.mean(accuracies)


def train_loop(train_loader, model, loss, optimizer, is_visualize=False):
    loss_history = []
    for x_batch, y_batch in train_loader:
        # visualize training data
        if is_visualize:
            train_x_mean, train_x_std = preprocess(x_batch, y_batch)
            visualize_data(train_x_mean, train_x_std, y_batch)
        logging.debug(f'input batch shape is {x_batch.shape}')
        logging.debug(f'output batch shape is {y_batch.shape}')
        # **[window_size, batch_size, channel]
        x_batch = torch.transpose(x_batch, 0, 1)
        logging.debug(f'transposed input shape is {x_batch.shape}')
        # training process
        optimizer.zero_grad()
        output = model(x_batch)
        # train loss
        logging.debug(f'y_batch shape is {y_batch.shape}')
        train_loss = -loss(output, y_batch)
        train_loss.backward()
        optimizer.step()
        loss_history.append(train_loss.item())
    train_loss = np.mean(loss_history)
    logging.info(f'Training loss: {train_loss}')
    return train_loss


def preprocess(train_x, train_y):
    # batch_size, window_size, channel_number
    logging.info(train_x.shape)
    # batch_size, mean_value_of_window for each channel
    train_x_mean = torch.mean(train_x, dim=1)
    train_x_std = torch.std(train_x, dim=1)
    for i in range(0, train_x_mean.shape[1]):
        visualize_data(train_x_mean[:, i], train_x_std[:, i], train_y[:, 0])
        visualize_data(train_x_mean[:, i], train_x_std[:, i], train_y[:, 1])
    logging.info(
        f'train_x_mean shape is {train_x_mean.shape}; and train_x_std shape is {train_x_std.shape}')
    return train_x_mean, train_x_std


def visualize_data(train_x_mean, train_x_std, train_y):
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 3))
        # Plot training data as black stars
        ax.errorbar(train_x_mean.numpy(), train_y, xerr=train_x_std, fmt='k*')
        ax.set_ylim([-1, 1])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()


def train_over_dataloader(train_loader, model, loss, optimizer, print_period=None):
    loss_history = []
    for idx, (features, labels) in enumerate(train_loader):
        # check NAN
        if torch.isnan(features).any() or torch.isnan(labels).any():
            logging.error('NAN happens')
            continue
        # **[window_size, batch_size, channel]
        features = torch.transpose(features, 0, 1).type(torch.FloatTensor)
        optimizer.zero_grad()
        output = model(features)
        # train loss
        train_loss = -loss(output, labels)
        logging.debug(f'output is {output.mean}')
        logging.debug(f'labels are {labels}')
        loss_history.append(train_loss.item())
        train_loss.backward(retain_graph=True)
        optimizer.step()
        if print_period is None:
            continue
        if idx > 0 and idx % print_period == 0:
            logging.info(f'   Step: {idx}, training loss: {np.mean(loss_history[-print_period:])}')

    mean_train_loss = np.mean(loss_history)
    logging.info(f'Training loss: {mean_train_loss}')
    return mean_train_loss


def valid_loop(valid_loader, model, likelihood, loss):
    loss_history = []
    for x_batch, y_batch in valid_loader:
        # **[window_size, batch_size, channel]
        x_batch = torch.transpose(x_batch, 0, 1)
        # cross validation process
        pred = likelihood(model(x_batch))
        # validation loss
        valid_loss = -loss(pred, y_batch)
        logging.info(f'Validation loss: {valid_loss}')
        loss_history.append(valid_loss.item())
    mean_valid_loss = np.mean(loss_history)
    logging.info(f'Validation loss: {mean_valid_loss}')
    return mean_valid_loss


def valid_and_trace(valid_loader, model, likelihood, loss, use_cuda=False, analyzer=None):
    loss_history = []
    # loss_info_history = []
    valid_accuracy = []
    for i, (X, y) in enumerate(valid_loader):
        X = torch.transpose(X, 0, 1).type(torch.FloatTensor)
        pred = likelihood(model(X))
        valid_loss = -loss(pred, y)
        mean = pred.mean
        logging.debug(f'validation batch {i} mean is {mean[0,:]}')
        loss_history.append(valid_loss.item())
        # TODO(SHU): add a loss wrapper
        # accuracy
        accuracy(y, mean, valid_accuracy)
        if analyzer is not None:
            analyzer.process(X, y, pred)

    valid_loss = np.mean(loss_history)
    logging.info(f'Validation loss: {valid_loss}.')
    logging.info(f'Validation accuracy = {np.mean(valid_accuracy)}')

    return valid_loss, np.mean(valid_accuracy)


def train(args, train_loader, valid_loader, total_train_number,
          print_period=None, early_stop=None, save_mode=0):
    timestr = time.strftime('%Y%m%d-%H%M%S')
    batch_size = args.batch_size
    num_inducing_point = args.num_inducing_point
    kernel_dim = args.kernel_dim
    use_cuda = args.use_cuda
    lr = args.lr
    epochs = args.epochs

    # init inducing points
    step_size = int(max(batch_size / num_inducing_point, 1))
    logging.info(f'step size is: {step_size}')
    inducing_point_num = torch.arange(0, batch_size, step=step_size)
    logging.info(f'inducing point indices are {inducing_point_num}')
    logging.info(train_loader)
    for idx, (features, labels) in enumerate(train_loader):
        features = torch.transpose(features, 0, 1).type(torch.FloatTensor)
        inducing_points = features[:, inducing_point_num, :].unsqueeze(0)
        # extend inducing points
        inducing_points = torch.cat((inducing_points, inducing_points), 0)
        # save inducing point for reload model
        np.save(os.path.join(args.validation_data_path, 'inducing_points.npy'), inducing_points)
        break
    # for saving on-line model
    for idx, (test_features, test_labels) in enumerate(valid_loader):
        test_features = torch.transpose(test_features, 0, 1).type(torch.FloatTensor)
        break

    # likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=OUTPUT_DIM)

    # encoder
    encoder_net_model = Encoder(u_dim=INPUT_DIM, kernel_dim=kernel_dim)

    # model
    model = GPModel(inducing_points=inducing_points,
                    encoder_net_model=encoder_net_model, num_tasks=OUTPUT_DIM)

    # optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    loss = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=total_train_number)

    # adjust learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                  verbose=False, threshold=0.0001, threshold_mode='rel',
                                  cooldown=0, min_lr=0.0, eps=1e-08)

    best_valid_loss = float('+inf')
    num_epoch_valid_loss_not_decreasing = 0
    # loss curve
    train_losses = []
    # train_accuracies = []
    valid_losses = []
    # valid_accuracies = []
    for epoch in range(1, epochs + 1):
        # running_loss = 0.0
        if use_cuda:
            likelihood.train().cuda
            model.train().cuda
        else:
            likelihood.train()
            model.train()
        train_loss = train_loop(train_loader, model, loss, optimizer)

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            valid_loss = valid_loop(valid_loader, model, likelihood, loss)
            # not calculating train accuracy for now
            # train_accuracy = result_tracing(train_loader, model, likelihood)
        scheduler.step(train_loss)
        if epoch == 10:
            gpytorch.settings.tridiagonal_jitter(1e-4)
        # loss and accuracy for each epoch
        train_losses.append(train_loss)
        # not calculating train accuracy for now
        # train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        # not calculating train accuracy for now
        # valid_accuracies.append(valid_accuracy)

        # Determine if valid_loss is getting better and if early_stop is needed.
        is_better_model = False
        if valid_loss < best_valid_loss:
            logging.info(
                f'****** current valid loss {valid_loss} is '
                + f'less then best valid loss {best_valid_loss}')
            num_epoch_valid_loss_not_decreasing = 0
            best_valid_loss = valid_loss
            is_better_model = True
        else:
            num_epoch_valid_loss_not_decreasing += 1
            logging.info(
                '****** number of valid loss not decreasing epoch is: '
                + f'{num_epoch_valid_loss_not_decreasing} ')
            # Early stop if enabled and met the criterion
            if early_stop == num_epoch_valid_loss_not_decreasing:
                logging.info('Reached early-stopping criterion. Stop training.')
                logging.info('Best validation loss = {}'.format(best_valid_loss))
                break

        # Save model according to the specified mode.
        online_model = os.path.join(args.gp_model_path, timestr, 'gp_model.pt')
        offline_model = os.path.join(args.gp_model_path, timestr, 'gp_model.pth')
        online_epoch_model = os.path.join(args.gp_model_path, timestr,
                                          'model_epoch{}_valloss{:.6f}.pt'.format(epoch,
                                                                                  valid_loss))
        offline_epoch_model = os.path.join(args.gp_model_path, timestr,
                                           'model_epoch{}_valloss{:.6f}.pth'.format(epoch,
                                                                                    valid_loss))
        if save_mode == 0:
            # save best model
            if is_better_model:
                save_model_torch_script(model, likelihood, test_features, online_model)
                save_model_state_dict(model, likelihood, offline_model)
        elif save_mode == 1:
            # save all better models
            if is_better_model:
                save_model_torch_script(model, likelihood, test_features, online_epoch_model)
                save_model_state_dict(model, likelihood, offline_epoch_model)
        elif save_mode == 2:
            # save all model
            save_model_torch_script(model, likelihood, test_features, online_epoch_model)
            save_model_state_dict(model, likelihood, offline_epoch_model)

    # plot loss curve
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_xlabel('epoch', fontdict={'size': 12})
    ax1.set_ylabel('loss', fontdict={'size': 12})
    ax1.plot(train_losses, label="training loss")
    ax1.plot(valid_losses, 'r.', label="validation loss")
    plt.legend(fontsize=12, numpoints=5, frameon=False)
    plt.title("Losses comparison")
    plt.grid(True)
    # # plot accuracy curve
    # TODO(Shu) Need to update
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.set_xlabel('epoch', fontdict={'size': 12})
    # ax2.set_ylabel('accuracy', fontdict={'size': 12})
    # ax2.plot(train_accuracies, label=f"training accuracy")
    # ax2.plot(valid_accuracies, label=f"validation accuracy")
    # plt.legend(fontsize=12, numpoints=5, frameon=False)
    # plt.grid(True)
    plt.savefig(args.gp_model_path + timestr + "loss_and_accuracy_curve.png")
    plt.show()

    return model, likelihood


class MeanVarModelWrapper(nn.Module):
    '''for online model saving'''

    def __init__(self, gp, likelihood):
        super().__init__()
        self.gp = gp
        self.likelihood = likelihood

    def forward(self, x):
        output_dist = self.likelihood(self.gp(x))
        return output_dist.mean, output_dist.variance


def save_model_torch_script(model, likelihood, test_features, file_name):
    '''save to TorchScript'''
    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    wrapped_model = MeanVarModelWrapper(model, likelihood)
    with gpytorch.settings.trace_mode(), torch.no_grad():
        pred = wrapped_model(test_features)  # Compute cache
        traced_model = torch.jit.trace(wrapped_model, test_features, check_trace=False)
        logging.info(f'saving model: {file_name}')
        logging.debug(f'pred is: {pred}')
    traced_model.save(file_name)


def save_model_state_dict(model, likelihood, file_name):
    '''save as state_dict'''
    model.eval()
    likelihood.eval()
    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    # state_dict = model.state_dict()
    logging.info(f'saving model state dict: {file_name}')
    torch.save([model.state_dict(), likelihood.state_dict()], file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument(
        '-t',
        '--training_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/gp_regression/testdata/train")
    parser.add_argument(
        '-v',
        '--validation_data_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/gp_regression/testdata/test")
    parser.add_argument(
        '--gp_model_path',
        type=str,
        default="/fuel/fueling/control/dynamic_model_2_0/gp_regression/gp_model_output")

    # model parameters
    parser.add_argument('-ni', '--num_inducing_point', type=int, default=1)
    parser.add_argument('--kernel_dim', type=int, default=20)
    # optimizer parameters
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-b', '--batch_size', type=int, default=1)

    # argument to use cuda or not for training
    parser.add_argument('--use_cuda', type=bool, default=False)
    args = parser.parse_args()

    # setup data-loader
    train_dataset = DynamicModelDataset(args.training_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    total_train_number = train_dataset.get_len()
    logging.info(f'total_train_number is {total_train_number}')

    valid_dataset = DynamicModelDataset(args.validation_data_path)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    for i, (X, y) in enumerate(train_loader):
        logging.info(
            f'training data batch: {i}, input size is {X.size()}, output size is {y.size()}')

    for i, (X, y) in enumerate(valid_loader):
        logging.info(
            f'validation data batch: {i}, input size is {X.size()}, output size is {y.size()}')

    train(args, train_loader, valid_loader, total_train_number)
    # python ./fueling/control/dynamic_model_2_0/gp_regression/train.py
    # -t /fuel/fueling/control/dynamic_model_2_0/testdata/0417/train
    # -v /fuel/fueling/control/dynamic_model_2_0/testdata/0417/validation
    # -ni 128
    # -e 300
