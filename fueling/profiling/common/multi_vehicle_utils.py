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
    # /apollo/modules/data/fuel/testdata/profiling/multi_job/genanrated/apollo/
    # 2019-11-08-15-36-49/Mkz7/Lon_Lat_Controller/Sim_Test-2019-05-01/20190501110414/
    controller_pos = target_dir.find('Controller')
    controller_dir = target_dir[:controller_pos]

    vehicle_param_dir = '/'.join(controller_dir.split('/')[:-1])
    return get_vehicle_param(vehicle_param_dir)
