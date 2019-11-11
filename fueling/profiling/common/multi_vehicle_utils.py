#!/usr/bin/env python
""" utils for multiple vehicles """
import os
import time

import numpy as np

import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils


def get_vehicle(path):
    return [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]


def get_vehicle_param(folder_dir):
    vehicle_para_conf_filename = 'vehicle_param.pb.txt'
    conf_file = os.path.join(folder_dir, vehicle_para_conf_filename)
    VEHICLE_PARAM_CONF = proto_utils.get_pb_from_text_file(
        conf_file, vehicle_config_pb2.VehicleConfig())
    return VEHICLE_PARAM_CONF.vehicle_param


def get_vehicle_param_by_target(target_dir):
    # target dir like
    # /apollo/2019-11-08-15-36-49/Mkz7/Lon_Lat_Controller/Sim_Test-2019-05-01/20190501110414/
    vehicle_param_dir_splited = target_dir.split('/')
    vehicle_param_dir = '/{}/{}/{}'.format(
        vehicle_param_dir_splited[1], vehicle_param_dir_splited[2], vehicle_param_dir_splited[3])
    return get_vehicle_param(vehicle_param_dir)
