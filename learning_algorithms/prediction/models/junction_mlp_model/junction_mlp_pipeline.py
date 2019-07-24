###############################################################################
# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import argparse
import logging
import os

import h5py
import numpy as np
import torch

from learning_algorithms.prediction.models.junction_mlp_model.junction_mlp_model import *
from learning_algorithms.utilities.train_utils import *


dim_input = 114
dim_output = 12

def load_h5(filename):
    """Load the data from h5 file to the format of numpy"""
    if not os.path.exists(filename):
        logging.error("file: {}, does not exist".format(filename))
        return None
    if os.path.splitext(filename)[1] != ".h5":
        logging.error("file: {} is not an hdf5 file".format(filename))
        return None
    samples = dict()
    h5_file = h5py.File(filename, "r")
    for key in h5_file.keys():
        samples[key] = h5_file[key][:]
    return samples["data"]

def data_preprocessing(data):
    """Preprocessing"""
    X = data[:, :dim_input]
    Y = data[:, -dim_output:]
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def do_training(source_save_paths):
    """Run training job"""
    logging.info("Start training job with paths: {}".format(source_save_paths))

    source_path, save_dir_path = source_save_paths 

    data = load_h5(source_path)
    if not data:
        logging.error("Failed to load data from {}".format(source_path))
        return

    logging.info("Data load success, with data shape: {}".format(str(data.shape)))
    train_data, test_data = train_test_split(data, test_size=0.2)
    X_train, Y_train = data_preprocessing(train_data)
    X_test, Y_test = data_preprocessing(test_data)

    logging.info(X_train.shape)

    # Model and training setup
    model = JunctionMLPModel(dim_input)
    loss = JunctionMLPLoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=2, min_lr=1e-8, verbose=True, mode="min")
    epochs = 20

    # CUDA setup:
    if torch.cuda.is_available():
       logging.info("Using CUDA to speed up training.")
       model.cuda()
       X_train = X_train.cuda()
       Y_train = Y_train.cuda()
       X_test = X_test.cuda()
       Y_test = Y_test.cuda()
    else:
       logging.info("Not using CUDA.")

    # Model training
    model = train_valid_vanilla(X_train, Y_train, X_test, Y_test, model, loss,
        optimizer, scheduler, epochs, "junction_mlp_model.pt", train_batch=1024)
    traced_script_module = torch.jit.trace(model, X_train[0:1])
    traced_script_module.save(os.path.join(save_dir_path, "junction_mlp_model.pt"))

    logging.info("Done with training job")


if __name__ == "__main__":

    # data parser:
    parser = argparse.ArgumentParser(
        description="semantic_map model training pipeline")

    parser.add_argument("--data", type=str, help="training data filename")
    parser.add_argument("-s", "--savepath", type=str, default="./",
                        help="Specify the directory to save trained models.")

    args = parser.parse_args()

    do_training((args.data, args.savepath))

