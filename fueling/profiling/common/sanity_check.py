#!/usr/bin/env python
import cgi
import collections
import os
import math
import sys

import google.protobuf.text_format as text_format

if sys.version_info[0] >= 3:
    from cyber_py3.record import RecordReader
else:
    from cyber_py.record import RecordReader
import modules.common.configs.proto.vehicle_config_pb2 as vehicle_config_pb2

import fueling.common.email_utils as email_utils
import fueling.common.logging as logging
import fueling.common.proto_utils as proto_utils
import fueling.common.record_utils as record_utils
import fueling.profiling.common.multi_vehicle_utils as multi_vehicle_utils


def list_records(path):
    logging.info("in list_records:%s" % path)
    records = []
    for (dirpath, _, filenames) in os.walk(path):
        logging.info('filenames: %s' % filenames)
        logging.info('dirpath %s' % dirpath)
        for filename in filenames:
            end_file = os.path.join(dirpath, filename)
            logging.info("end_files: %s" % end_file)
            if record_utils.is_record_file(end_file):
                records.append(end_file)
    return records


def missing_file(path, conf_pb):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    logging.info("vehicles %s" % vehicles)
    for vehicle in vehicles:
        # config file
        conf = os.path.join(
            path, vehicle, conf_pb)
        logging.info("vehicles conf %s" % conf)
        if os.path.exists(conf) is False:
            logging.error('Missing configuration file in %s' % vehicle)
            return True
        # record file
        logging.info("list of records:" %
                     list_records(os.path.join(path, vehicle)))
        if len(list_records(os.path.join(path, vehicle))) == 0:
            logging.error('No record files in %s' % vehicle)
            return True
    return False


def parse_error(path, conf_pb):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    pb_value = vehicle_config_pb2.VehicleConfig()
    for vehicle in vehicles:
        conf = os.path.join(
            path, vehicle, conf_pb)
        try:
            proto_utils.get_pb_from_text_file(conf, pb_value)
            return False
        except text_format.ParseError:
            logging.error(
                'Error: Cannot parse %s as binary or text proto' % conf)
            return True


def check_vehicle_id(conf):
    vehicle_id = conf.vehicle_param.vehicle_id
    if vehicle_id.vin or vehicle_id.plate or vehicle_id.other_unique_id:
        return True
    logging.error("Error: No vehicle ID")
    return False


def missing_field(path, conf_pb):
    vehicles = multi_vehicle_utils.get_vehicle(path)
    logging.info("vehicles in missing field: %s" % vehicles)
    for vehicle in vehicles:
        conf_file = os.path.join(
            path, vehicle, conf_pb)
        logging.info("conf_file: %s" % conf_file)
        # reset for each vehicle to avoid overwrited
        pb_value = vehicle_config_pb2.VehicleConfig()
        conf = proto_utils.get_pb_from_text_file(conf_file, pb_value)
        logging.info("vehicles conf %s" % conf)
        if not check_vehicle_id(conf):
            return True
        # required field
        fields = [conf.vehicle_param.wheel_base]
        # for value in conf.vehicle_param:
        # has field is always true since a default value is given
        for field in fields:
            if math.isnan(field):
                return True
    return False


def missing_message_data(path, channels):
    for record in list_records(path):
        logging.info("reading records %s" % record)
        reader = RecordReader(record)
        for channel in channels:
            logging.info("has %d messages" % reader.get_messagenumber(channel))
            if reader.get_messagenumber(channel) == 0:
                return True
    return False


def sanity_check(input_folder, conf_pb, channels, job_owner, job_id):
    err_msg = None
    if missing_file(input_folder, conf_pb):
        err_msg = "One or more files are missing in %s" % input_folder
    elif parse_error(input_folder, conf_pb):
        err_msg = "Config file cannot be parsed in %s" % input_folder
    elif missing_field(input_folder, conf_pb):
        err_msg = "One or more fields are missing in config file %s" % input_folder
    elif missing_message_data(input_folder, channels):
        err_msg = "Messages are missing in records of %s" % input_folder
    else:
        logging.info("%s Passed sanity check." % input_folder)
        return True

    logging.error(err_msg)
    return False
