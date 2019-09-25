#!/usr/bin/env python

import torch

import fueling.common.learning.network_utils as network_utils
import fueling.common.logging as logging


class NeuralNetworkTorch(object):
    """ neural network for auto calibration """

    def __init__(self, layer):
        last_layer_nonlinear = False
        self.net = network_utils.generate_mlp(layer, last_layer_nonlinear).double()
        self.net.apply(self.init_weights)

    def train(self, X_train, Y_train, X_test, Y_test, alpha=0.01, w_lambda=0.01, num_epoch=10000):
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_test = torch.from_numpy(X_test)
        Y_test = torch.from_numpy(Y_test)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=alpha, weight_decay=w_lambda)

        # Training mode.
        self.net.train()
        train_cost = 0
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            outputs = self.net(X_train)
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()
            train_cost = loss.item()
            if epoch % 100 == 0:
                logging.info("loss after epoch {}: {}".format(epoch, train_cost))

        # Evaluation mode.
        test_cost = 0
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(X_test)
            loss = criterion(outputs, Y_test)
            test_cost = loss.item()

        return self.net.parameters(), train_cost, test_cost

    def predict(self, X_data):
        return self.net(torch.from_numpy(X_data)).detach().numpy()

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
