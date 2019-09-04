#!/usr/bin/env python

import argparse
import glob
import os
import pickle

import colored_glog as glog
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import pyro
import pyro.contrib.gp as gp
import torch
import torch.nn as nn

from fueling.control.dynamic_model.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model.gp_regression.train import train_gp, DeepEncodingNet


def test_gp(args, dataset, GaussianProcess):
    """Load GP model params from files and make predictions and testset"""
    input_data, gt_data = dataset.get_test_data()

    for sub_dir in os.scandir(args.gp_model_path):
        glog.info("************Loading GP model from {}".format(sub_dir))
        fnet_dict = torch.load(os.path.join(sub_dir, "fnet.p"))
        lik_dict = torch.load(os.path.join(sub_dir, "likelihood.p"))
        kernel_dict = torch.load(os.path.join(sub_dir, "kernel.p"))
        gp_dict = torch.load(os.path.join(sub_dir, "gp_f.p"))

        deep_encoding_net = DeepEncodingNet(args, input_data.shape[2], args.kernel_dim)
        deep_encoding_net.load_state_dict(fnet_dict)

        def _encoded_feature(original_feature):
            return pyro.module("DeepEncodingNet", deep_encoding_net)(original_feature)

        Xu = input_data[torch.arange(0, input_data.shape[0],
                        step=int(max(input_data.shape[0] / args.num_inducing_point, 1))).long()]
        likelihood = gp.likelihoods.Gaussian(variance=torch.ones(2, 1))
        likelihood.load_state_dict(lik_dict)

        kernelization = gp.kernels.Matern52(input_dim=args.kernel_dim,
                                            lengthscale=torch.ones(args.kernel_dim))
        kernel = gp.kernels.Warping(kernelization, iwarping_fn=_encoded_feature)
        kernel.load_state_dict(kernel_dict)
        gp_f = gp.models.VariationalSparseGP(input_data, torch.ones(2, input_data.shape[0]),
                                             kernel, Xu, num_data=input_data.shape[0],
                                             likelihood=likelihood, mean_function=None,
                                             whiten=True, jitter=1e-4)
        gp_f.load_state_dict(gp_dict)
        gp_model = GaussianProcess(args, gp_f, dataset)
        predicted_data = torch.zeros(0, gt_data.shape[1])

        for i in range(len(input_data)):
            glog.debug("Input Dim {}".format((input_data[i].unsqueeze(0)).size()))
            glog.debug("Label Dim {}".format(gt_data[i].size()))
            predicted_mean, predicted_var = gp_model.gp_f(input_data[i].unsqueeze(0), full_cov=True)
            predicted_data = torch.cat((predicted_data,
                                        torch.tensor([predicted_mean[0],
                                                     predicted_mean[1]]).unsqueeze(0)), 0)
            glog.info("predicted mean:{}".format(predicted_mean))
            glog.info("predicted variance:{}".format(predicted_var))
            glog.info("ground-truth residual error:{}".format(gt_data[i]))
        
        input_data = input_data.numpy()

        '''
        # Input feature visualization
        for i in range(len(input_data)):
            plt.figure(figsize=(4, 3))
            plt.title("Dataset Visualization")
            plt.plot(input_data[i, :, 0], 'o', color='black')
            plt.plot(input_data[i, :, 1], 'o', color='grey')
            plt.plot(input_data[i, :, 2], 'o', color='green')
            plt.plot(input_data[i, :, 3], 'o', color='red')
            plt.plot(input_data[i, :, 4], 'o', color='blue')
            plt.show()
        '''

        glog.debug("gt_data: {}".format(gt_data))
        glog.debug("predicted_data: {}".format(predicted_data))
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(gt_data, predicted_data))
        glog.info("RMSE:{}".format(loss))
        gt_data = gt_data.numpy()
        predicted_data = predicted_data.numpy()
        plt.figure(figsize=(4, 3))
        plt.title("Result Visualization")
        plt.plot(gt_data[: , 0], gt_data[: , 1], 'o', color='blue')
        plt.plot(predicted_data[: , 0], predicted_data[: , 1], 'o', color='red')
        plt.plot([gt_data[: , 0], predicted_data[: , 0]], [gt_data[: , 1],
                 predicted_data[: , 1]], 'g:')
        plt.show()
