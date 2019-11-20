#!/usr/bin/env python

import argparse
import os
import pickle

from datetime import datetime

import torch
import asyncio
import websockets
from fueling.control.dynamic_model.gp_regression.dataset import GPDataSet
from fueling.control.dynamic_model.gp_regression.dreamview_server import load_gp, run_gp
from fueling.control.dynamic_model.gp_regression.evaluation import test_gp
from fueling.control.dynamic_model.gp_regression.gaussian_process import GaussianProcess
from fueling.control.dynamic_model.gp_regression.train import train_gp


def launch(args):
    # tasks to launch
    args.train_gp = False
    args.test_gp = False

    dataset = GPDataSet(args)
    if args.train_gp:
        # train Gaussian process model
        train_gp(args, dataset, GaussianProcess)
    if args.test_gp:
        # test Gaussian process model
        test_gp(args, dataset, GaussianProcess)


async def run_model(websocket, path):
    #dataset = GPDataSet(args)
    input_string = await websocket.recv()
    print(f"< {input_string}")
    input_data = torch.zeros(1, 100, 6)
    # load the trained GP model
    #gp_f = load_gp(args, dataset)
    # keep being called by the web-socket and return predicted mean and var
    #(predicted_mean, predicted_var) = run_gp(gp_f, input_data)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_string = f"Send back {input_string} at time {current_time}!"

    await websocket.send(output_string)
    print(f"> {output_string}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP')
    # paths
    parser.add_argument(
        '--training_data_path',
        type=str,
        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/dataset/tmp/training")
    parser.add_argument(
        '--testing_data_path',
        type=str,
        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/dataset/tmp/testing")
    parser.add_argument(
        '--gp_model_path',
        type=str,
        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/gp_model")
    parser.add_argument(
        '--eval_result_path',
        type=str,
        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/results")
    parser.add_argument(
        '--online_gp_model_path',
        type=str,
        default="/apollo/modules/data/fuel/testdata/control/gaussian_process/gp_model/20191004-130454")

    # model parameters
    parser.add_argument('--delta_t', type=float, default=0.01)
    parser.add_argument('--Delta_t', type=float, default=1)
    parser.add_argument('--num_inducing_point', type=int, default=10)
    parser.add_argument('--kernel_dim', type=int, default=20)

    # optimizer parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--compute_normalize_factors', type=bool, default=True)
    parser.add_argument('--compare', type=str, default="model")

    args = parser.parse_args()
    launch(args)

    start_server = websockets.serve(run_model, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
