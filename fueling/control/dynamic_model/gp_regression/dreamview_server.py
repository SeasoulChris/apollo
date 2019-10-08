#!/usr/bin/env python

import argparse
import glob
import os
import pickle

import numpy as np
import pyro
import pyro.contrib.gp as gp
import torch

from fueling.control.dynamic_model.gp_regression.train import DeepEncodingNet
import fueling.common.logging as logging


def load_gp(args, dataset):
    """
    Load GP model params from files
    Input: the pre-defined arguments and dataset
    Output: an instance of pyro VariationalSparseGP loaded with pretrained params
    """
    input_data, gt_data = dataset.get_test_data()
    fnet_dict = torch.load(os.path.join(args.online_gp_model_path, "fnet.p"))
    lik_dict = torch.load(os.path.join(args.online_gp_model_path, "likelihood.p"))
    kernel_dict = torch.load(os.path.join(args.online_gp_model_path, "kernel.p"))
    gp_dict = torch.load(os.path.join(args.online_gp_model_path, "gp_f.p"))

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
    return gp_f

def run_gp(gp_f, input_data):
    """
    Run the main logic of the web-socket server
    Input: an array of last k (default k = 100) frames' input features (default dim = 6)
    Output: the gp_regression output: mean and variance of the residual prediction
    """
    logging.debug("Input Dim {}".format(input_data.size()))
    predicted_mean, predicted_var = gp_f(input_data, full_cov=True)
    logging.info("predicted mean:{}".format(predicted_mean))
    logging.info("predicted variance:{}".format(predicted_var))
    return (predicted_mean, predicted_var)