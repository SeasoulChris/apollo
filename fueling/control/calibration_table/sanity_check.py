#!/usr/bin/env python
import os
import math

import colored_glog as glog
import google.protobuf.text_format as text_format

from cyber_py.record import RecordReader
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.control.common.multi_vehicle_utils as multi_vehicle_utils

# could be a list
ConfFile = 'vehicle_param.pb.txt'
CHANNELS = {record_utils.CHASSIS_CHANNEL, record_utils.LOCALIZATION_CHANNEL}


def list_records(path):
    records = []
    for (dirpath, _, filenames) in os.walk(path):
        for filename in filenames:
            end_file = os.path.join(dirpath, filename)
            if record_utils.is_record_file(end_file):
                records.append(end_file)  # TODO: check
    return records


def missing_file(path):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    for vehicle in vehicles:
        # config file
        conf = os.path.join(path, vehicle, ConfFile)
        if os.path.exists(conf) is False:
            glog.error('Missing configuration file')
            return True
    # record file
    if len(list_records(path)) == 0:
        glog.error('No record files')
        return True
    return False


def parse_error(path):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    pb_value = vehicle_config_pb2.VehicleConfig()
    for vehicle in vehicles:
        conf = os.path.join(path, vehicle, ConfFile)
        try:
            proto_utils.get_pb_from_text_file(conf, pb_value)
            return False
        except text_format.ParseError:
            glog.error('Error: Cannot parse %s as binary or text proto' % conf)
            return True


def check_vehicle_id(conf):
    # print(conf.HasField('vehicle_id.other_unique_id'))
    vehicle_id = conf.vehicle_param.vehicle_id
    if vehicle_id.HasField('vin') or vehicle_id.HasField('plate') \
            or vehicle_id.HasField('other_unique_id'):
        return True
    glog.error("Error: No vehicle ID")
    return False


def missing_field(path):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    pb_value = vehicle_config_pb2.VehicleConfig()
    for vehicle in vehicles:
        conf_file = os.path.join(path, vehicle, ConfFile)
        conf = proto_utils.get_pb_from_text_file(conf_file, pb_value)
        glog.error(conf)
        if not check_vehicle_id(conf):
            return False
        # required field
        fields = [conf.vehicle_param.brake_deadzone,
                  conf.vehicle_param.throttle_deadzone,
                  #   conf.vehicle_param.max_acceleration,
                  conf.vehicle_param.max_deceleration]
        # for value in conf.vehicle_param:
        # has field is always true since a default value is given
        for field in fields:
            if math.isnan(field):
                return True
        return False


def missing_message_data(path, channels=CHANNELS):
    for record in list_records(path):
        reader = RecordReader(record)
        for channel in channels:
            if reader.get_messagenumber(channel) == 0:
                return True
    return False


def sanity_check(input_folder):
    if missing_file(input_folder):
        glog.error("One or more files are missing")
        return False
    elif parse_error(input_folder):
        glog.error("Confige file cannot be parsed")
        return False
    elif missing_field(input_folder):
        glog.error("One or more field is missing in Confige file")
        return False
    elif missing_message_data(input_folder):
        glog.error("Messages are missing in records")
        return False
    else:
        glog.info("Passed sanity check.")
        return True


if __name__ == '__main__':
    sanity_check('/apollo/modules/data/fuel/testdata/control/sourceData/SanityCheck')
