#!/usr/bin/env python
"""Training models"""

import os
import time

import numpy as np
import pyro
import pyro.contrib.gp as gp
import torch
import torch.nn as nn
import torch.nn.functional as Func

from fueling.control.dynamic_model_2_0.conf.model_conf import segment_index, feature_config
from fueling.control.dynamic_model_2_0.conf.model_conf import input_index, output_index
from fueling.control.dynamic_model_2_0.gp_regression.encoder import Encoder
from fueling.control.dynamic_model_2_0.gp_regression.predict import Predict
import fueling.common.logging as logging

# Default (x,y) residual error correction cycle is 1s;
# Default control/chassis command cycle is 0.01s;
# Every 100 frames Input Vector correspond to 1 frame of output.
INPUT_LENGTH = feature_config["DELTA_T"] / feature_config["delta_t"]  # Default 100


def preprocessing(args, dataset, gp):
    # TODO(Shu): Implement data preprocessing module
    logging.info("End of preprocessing")


def save_gp(args, gp_model, feature, encoder):
    """Save the learned models for Gaussian process"""
    timestr = time.strftime('%Y%m%d-%H%M%S')
    model_saving_path = os.path.join(args.gp_model_path, timestr)
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path)

    gp_model.eval()
    encoder.eval()

    torch.save(gp_model.gp_f.state_dict(), os.path.join(model_saving_path, "gp_f.p"))
    torch.save(gp_model.gp_f.kernel.state_dict(), os.path.join(model_saving_path, "kernel.p"))
    torch.save(gp_model.gp_f.likelihood.state_dict(),
               os.path.join(model_saving_path, "likelihood.p"))
    torch.save(encoder.state_dict(), os.path.join(model_saving_path, "fnet.p"))
    predict_fn = Predict(gp_model.model, gp_model.guide)
    # predict_module = torch.jit.trace_module(predict_fn, {"forward": (feature,)}, check_trace=False)
    # # TypeError: guide() takes 1 positional argument but 2 were given
    # torch.jit.save(predict_module, '/tmp/reg_predict.pt')
    encoder_module = torch.jit.trace_module(encoder, {"forward": (feature,)}, check_trace=False)
    torch.jit.save(encoder_module, '/tmp/encoder_module.pt')


def train_gp(args, dataset, gp_class):
    """Train the dataset with Gaussian Process assumption"""
    feature, label = dataset.get_train_data()
    logging.info("************Input Dim: {}".format(feature.shape))
    logging.info("************Input Example: {}".format(feature[0]))
    logging.info("************Output Dim: {}".format(label.shape))
    logging.info("************Output Example: {}".format(label[0]))

    # Encode the original features to lower-dimensional feature with kernel size
    encoder = Encoder(args, feature.shape[2], args.kernel_dim)

    def _encoded_feature(original_feature):
        return pyro.module("Encoder", encoder)(original_feature)

    likelihood = gp.likelihoods.Gaussian(variance=0.1 * torch.ones(2, 1))
    kernelization = gp.kernels.Matern52(input_dim=args.kernel_dim,
                                        lengthscale=torch.ones(args.kernel_dim))
    # Kernelize the encoding features at lower-dimension
    kernel = gp.kernels.Warping(kernelization, iwarping_fn=_encoded_feature)
    # Define the inducing points of Gaussian Process
    Xu = feature[torch.arange(0, feature.shape[0],
                              step=int(max(feature.shape[0] / args.num_inducing_point, 1))).long()]
    # The Pyro core of Gaussian Process training through variational inference
    gp_f = gp.models.VariationalSparseGP(feature, label, kernel, Xu,
                                         num_data=feature.shape[0], likelihood=likelihood,
                                         mean_function=None, whiten=True, jitter=1e-3)
    # Instantiate a Gaussian Process object
    gp_instante = gp_class(args, gp_f, dataset)
    # args.mate = preprocessing(args, dataset, gp_instante)
    # Pyro Adam optimizer
    optimizer = pyro.optim.ClippedAdam({"lr": args.lr, "lrd": args.lr_decay})
    # svi: Stochastic variational inference
    # ELBO: Evidence Lower Boundary
    svi = pyro.infer.SVI(gp_instante.model, gp_instante.guide,
                         optimizer, pyro.infer.JitTrace_ELBO())

    logging.info("Start of training")
    gp_instante.set_data(feature, label)
    for epoch in range(1, args.epochs + 1):
        loss = svi.step()
        logging.info('Train Epoch: {:2d} \tLoss: {:.6f}'.format(epoch, loss))
        if epoch == 10:
            gp_instante.gp_f.jitter = 1e-4

    save_gp(args, gp_instante, feature, encoder)
