#!/usr/bin/env python
import cgi
import collections
import os
import math
import sys

import google.protobuf.text_format as text_format

from cyber_py3.record import RecordReader
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

import fueling.common.email_utils as email_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils


def list_records(path):
    logging.info(F'in list_records: {path}')
    records = []
    for (dirpath, _, filenames) in os.walk(path):
        logging.info(F'filenames: {filenames}')
        logging.info(F'dirpath: {dirpath}')
        for filename in filenames:
            end_file = os.path.join(dirpath, filename)
            logging.info(F'end_files: {end_file}')
            if record_utils.is_record_file(end_file) or record_utils.is_bag_file(end_file):
                records.append(end_file)
    return records


def missing_file(path, conf_pb):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    logging.info(F'vehicles: {vehicles}')
    for vehicle in vehicles:
        # config file
        conf = os.path.join(
            path, vehicle, conf_pb)
        logging.info(F'vehicles conf: {conf}')
        if os.path.exists(conf) is False:
            status = F'Missing configuration file in vehicle: {vehicle}'
            logging.error(status)
            return status
        # record file
        logging.info(F'list of records: {list_records(os.path.join(path, vehicle))}')
        if len(list_records(os.path.join(path, vehicle))) == 0:
            status = F'No record files in vehicle: {vehicle}'
            logging.error(status)
            return status
    return 'OK'


def parse_error(path, conf_pb):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    pb_value = vehicle_config_pb2.VehicleConfig()
    for vehicle in vehicles:
        conf = os.path.join(
            path, vehicle, conf_pb)
        try:
            proto_utils.get_pb_from_text_file(conf, pb_value)
            return 'OK'
        except text_format.ParseError:
            status = F'Cannot parse {conf} as binary or text proto'
            logging.error(status)
            return status


def check_vehicle_id(conf):
    vehicle_id = conf.vehicle_param.vehicle_id
    if vehicle_id.vin or vehicle_id.plate or vehicle_id.other_unique_id:
        return 'OK'
    status = 'No vehicle ID'
    logging.error(status)
    return status


def missing_field(path, conf_pb):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    logging.info(F'vehicles in missing field: {vehicles}')
    for vehicle in vehicles:
        conf_file = os.path.join(
            path, vehicle, conf_pb)
        logging.info(F'conf_file: {conf_file}')
        # reset for each vehicle to avoid overwrited
        pb_value = vehicle_config_pb2.VehicleConfig()
        conf = proto_utils.get_pb_from_text_file(conf_file, pb_value)
        logging.info(F'vehicles conf: {conf}')
        id_status = check_vehicle_id(conf)
        if id_status is not 'OK':
            status = id_status + F' in conf: {conf_pb} for vehicle: {vehicle}'
            return status
        # required field
        fields = ['wheel_base']
        # for value in conf.vehicle_param:
        # has field is always true since a default value is given
        for field in fields:
            field_value = getattr(conf.vehicle_param, field, float('NaN'))
            if math.isnan(field_value):
                status = F'Missing field: {field} in conf: {conf_pb} for vehicle: {vehicle}'
                return status
    return 'OK'


def missing_message_data(path, channels):
    for record in list_records(path):
        logging.info(F'reading records: {record}')
        reader = RecordReader(record)
        for channel in channels:
            logging.info(F'has {reader.get_messagenumber(channel)} messages')
            if reader.get_messagenumber(channel) == 0:
                status = F'Missing message channel: {channel} in record: {record}'
                return status
    return 'OK'


def sanity_check(input_folder, conf_pb, channels):
    def error_message(status, folder):
        message = (F'Sanity_Check: Failed; \n'
                   F'Detailed Reason: {status}; \n'
                   F'Data Directory: {folder}.')
        return message
    current_status = missing_file(input_folder, conf_pb)
    if current_status is not 'OK':
        status_msg = error_message(current_status, input_folder)
        return status_msg
    current_status = parse_error(input_folder, conf_pb)
    if current_status is not 'OK':
        status_msg = error_message(current_status, input_folder)
        return status_msg
    current_status = missing_field(input_folder, conf_pb)
    if current_status is not 'OK':
        status_msg = error_message(current_status, input_folder)
        return status_msg
    current_status = missing_message_data(input_folder, channels)
    if current_status is not 'OK':
        status_msg = error_message(current_status, input_folder)
        return status_msg
    status_msg = 'OK'
    return status_msg
