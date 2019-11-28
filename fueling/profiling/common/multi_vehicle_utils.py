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


def get_controller_dir_by_target(target_dir):
    # target dir like
    # /apollo/modules/data/fuel/testdata/profiling/multi_job_genanrated/apollo/
    # 2019-11-22-15-36-49/Mkz7/Lon_Lat_Controller/Sim_Test-2019-05-01/20190501110414/
    controller_pos = target_dir.find('Controller')
    return target_dir[:controller_pos]


def get_vehicle_by_target(target_dir):
    controller_dir = get_controller_dir_by_target(target_dir)
    return controller_dir.split('/')[-2]


def get_controller_by_target(target_dir):
    controller_dir = get_controller_dir_by_target(target_dir)

    return '{}Controller'.format(controller_dir.split('/')[-1])


def get_vehicle_param_by_target(target_dir):
    controller_dir = get_controller_dir_by_target(target_dir)

    vehicle_param_dir = '/'.join(controller_dir.split('/')[:-1])
    return get_vehicle_param(vehicle_param_dir)


def get_vehicle_by_task(task_dir):
    # task dir like
    # /apollo/modules/data/fuel/testdata/profiling/multi_job/Mkz7/Sim_Test-2019-05-01/20190501110414
    return task_dir.split('/')[-3]


def get_target_removed_controller(task_dir):
    # task dir like
    # [('Mkz7', '/mnt/bos/modules/control/profiling/apollo/2019
    # /Mkz7/Lon_Lat_Controller/Road_Test-2019-05-01/20190501110414'),...]
    dir_list = task_dir.split('/')
    controller_pos = [i for i,x in enumerate(dir_list) if x.find('Controller') != -1]
    del dir_list[controller_pos[0]]
    return ('/').join(dir_list)
